"""Check E7 encoding, search and CUDA metrics on a small SciFact manifest."""

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from csrv2_repro.common import RESULTS, load_config
from csrv2_repro.e7_profiling import load_rows


def write_config(path: Path, cfg: dict) -> None:
    """Write the flat settings and model tables used by the E7 pilot config."""
    lines = [
        f"{key} = {json.dumps(value)}" for key, value in cfg.items() if not isinstance(value, dict)
    ]
    for name, table in cfg.items():
        if isinstance(table, dict):
            lines.append(f"\n[{name}]")
            lines.extend(f"{key} = {json.dumps(value)}" for key, value in table.items())
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def verify_results(run_name: str) -> None:
    """Verify the expected timings, coverage and allocator metrics on CUDA."""
    out = RESULTS / "E7" / run_name
    encoding = json.loads((out / "encoding.json").read_text())["measurements"]
    for name in ("dense", "csrv2"):
        for role, count in (("query", 8), ("document", 24)):
            measurement = encoding[name][role]
            assert measurement["texts"] == measurement["profiled_texts"] == count
            assert measurement["profiled_batches"] == count // 4
            for key in ("seconds", "texts_per_second", "backbone_stream_ms_per_batch_mean"):
                assert math.isfinite(measurement[key]) and measurement[key] > 0, key
            head_ms = measurement["sae_stream_ms_per_batch_mean"]
            if name == "csrv2":
                assert math.isfinite(head_ms) and head_ms > 0
            else:
                assert head_ms is None

    calibration = encoding["csrv2"]["document"]["marker_calibration"]
    assert calibration["pairs"] == 20
    assert math.isfinite(calibration["overhead_pct"])
    assert calibration["recommended_stride"] == (16 if abs(calibration["overhead_pct"]) > 1 else 1)
    retrieval = json.loads((out / "retrieval_encoded.json").read_text())["measurements"]
    assert {row["method"] for row in retrieval} == {"qwen_dense", "mrl", "csrv2"}
    for row in retrieval:
        assert row["documents"] == 24 and row["queries"] == 8 and row["query_batch_size"] == 4
        assert row["corpus_bytes"] > 0 and row["ms_per_batch_mean"] > 0
        memory = row["cuda_memory"]
        before = memory["allocated_before_bytes"]
        peak = memory["peak_allocated_bytes"]
        assert peak >= before >= row["corpus_bytes"]
        assert memory["extra_peak_bytes"] == peak - before
    print(f"E7 smoke passed. Results: {out}\nMarker calibration: {json.dumps(calibration)}", flush=True)


def main() -> None:
    """Run encode and encoded search with the pilot checkpoint revisions."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("checkpoint_home", type=Path, nargs="?")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("this end-to-end smoke requires CUDA")
    cfg = load_config(args.config)
    rows = load_rows(cfg)
    subset = [row for row in rows if row["role"] == "query"][:8]
    subset += [row for row in rows if row["role"] == "document"][:24]
    assert len(subset) == 32
    with TemporaryDirectory(prefix="e7_smoke_") as temp:
        scratch = Path(temp)
        manifest = scratch / "manifest.jsonl"
        manifest.write_text("".join(json.dumps(row) + "\n" for row in subset), encoding="utf-8")
        cfg.update(
            run_name=scratch.name,
            manifest=str(manifest),
            encoding_batch_size=4,
            encoding_warmup=2,
            encoding_k=2,
            encoding_profile_stride=1,
            encoding_calibration_pairs=20,
            ks=[2],
            query_batch_size=4,
            corpus_block_size=12,
            search_warmup=2,
            search_rounds=3,
            profile_cuda=False,
        )
        config = scratch / "smoke.toml"
        write_config(config, cfg)
        # Keep model loading offline. The checkpoint files must already be downloaded.
        env = dict(os.environ, HF_HUB_OFFLINE="1")
        if args.checkpoint_home is not None:
            checkpoint_home = args.checkpoint_home.resolve()
            hub_cache = str(checkpoint_home / "hub")
            env.update(
                HF_HOME=str(checkpoint_home),
                HF_HUB_CACHE=hub_cache,
                HUGGINGFACE_HUB_CACHE=hub_cache,
                TRANSFORMERS_CACHE=hub_cache,
                SENTENCE_TRANSFORMERS_HOME=hub_cache,
            )
            print(f"Reading downloaded checkpoints from {hub_cache}", flush=True)
        command = [sys.executable, "-m", "csrv2_repro.e7_profiling"]
        subprocess.run([*command, "encode", "--config", str(config)], env=env, check=True)
        subprocess.run(
            [*command, "search", "--config", str(config), "--source", "encoded"], env=env, check=True
        )
        verify_results(cfg["run_name"])


if __name__ == "__main__":
    main()
