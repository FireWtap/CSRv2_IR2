"""Encode manifest training/validation queries missing from the shared E1 dense cache.

Prepare pinned inputs on the login node with --prepare-only. Run GPU extraction using
--prepared <path printed by preparation>. Embeddings stay on scratch; results contain provenance.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
from pathlib import Path

import numpy as np

from csrv2_repro.common import (
    OUTPUTS,
    REPO_ROOT,
    RESULTS,
    load_config,
    provenance,
    resolve_dtype,
    upstream_prompts,
    write_json,
)
from csrv2_repro.splits import digest, load_manifest, load_source, load_source_qrels, select_qrels

logger = logging.getLogger(__name__)


def prepare_inputs(cfg: dict, manifest: dict, dense_cfg: dict) -> dict:
    if set(cfg["partitions"]) - {"train", "validation"}:
        raise ValueError("E3 extraction only accepts train/validation partitions")
    parts = []
    for name in cfg["tasks"]:
        entry = manifest["tasks"][name]
        qrels = load_source_qrels(entry["source"])
        queries = next(iter(load_source(entry["source"], "queries").values()))
        texts = {}
        for row in queries:
            qid = str(row["_id"])
            if qid in texts:
                raise ValueError(f"Duplicate source query ID: {name}/{qid}")
            texts[qid] = row["text"]
        for partition in cfg["partitions"]:
            selected = select_qrels(entry, qrels, partition)
            ids = list(selected)
            if not set(ids) <= texts.keys():
                raise ValueError(f"Missing query text: {name}/{partition}")
            parts.append(
                {
                    "task": name,
                    "partition": partition,
                    "query_ids": ids,
                    "texts": [texts[q] for q in ids],
                }
            )
    prepared = {
        "manifest_sha256": manifest["sha256"],
        "config": cfg,
        "dense_config": dense_cfg,
        "prompts": upstream_prompts(),
        "parts": parts,
    }
    prepared["sha256"] = digest(prepared)
    return prepared


def validate_prepared(prepared: dict, cfg: dict, manifest: dict, dense_cfg: dict) -> None:
    if prepared["sha256"] != digest({k: v for k, v in prepared.items() if k != "sha256"}):
        raise ValueError("Prepared-input checksum mismatch")
    if (
        prepared["config"] != cfg
        or prepared["dense_config"] != dense_cfg
        or prepared["manifest_sha256"] != manifest["sha256"]
        or prepared["prompts"] != upstream_prompts()
    ):
        raise ValueError("Prepared inputs do not match the requested configuration")
    expected = {(t, p) for t in cfg["tasks"] for p in cfg["partitions"]}
    observed = [(p["task"], p["partition"]) for p in prepared["parts"]]
    if len(observed) != len(expected) or set(observed) != expected:
        raise ValueError("Prepared task/partition coverage mismatch")
    for part in prepared["parts"]:
        ids = manifest["tasks"][part["task"]]["partitions"][part["partition"]]["query_ids"]
        if part["query_ids"] != ids or len(part["texts"]) != len(ids):
            raise ValueError("Prepared query order does not match the frozen manifest")


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def encode_partition(
    encoder, part: dict, directory: Path, manifest_sha: str, prepared_sha: str
) -> dict:
    from mteb.encoder_interface import PromptType

    name, partition = part["task"], part["partition"]
    ids, texts = part["query_ids"], part["texts"]
    if not ids:
        return {"task": name, "partition": partition, "n_queries": 0, "status": "empty"}
    prompt = encoder._prompt(name, PromptType.query)
    cache_path = encoder._cache_path(texts, prompt, name, PromptType.query)
    cached = cache_path.exists()
    vectors = encoder.embed(texts, task_name=name, prompt_type=PromptType.query)
    dim = encoder.model.get_sentence_embedding_dimension()
    if vectors.shape != (len(ids), dim) or vectors.dtype != np.float32:
        raise ValueError(f"Unexpected embedding shape/dtype: {name}/{partition}")
    if not np.isfinite(vectors).all() or np.any(np.linalg.norm(vectors, axis=1) == 0):
        raise ValueError(f"Nonfinite/zero embedding: {name}/{partition}")
    record = {
        "task": name,
        "partition": partition,
        "n_queries": len(ids),
        "query_ids": ids,
        "texts_sha256": digest(texts),
        "manifest_sha256": manifest_sha,
        "prepared_sha256": prepared_sha,
        "model_tag": encoder.model_tag,
        "prompt_sha256": digest(prompt),
        "embedding_path": str(cache_path.resolve()),
        "embedding_sha256": file_sha256(cache_path),
        "shape": list(vectors.shape),
        "dtype": str(vectors.dtype),
    }
    path = directory / name / f"{partition}.json"
    write_json(path, record)
    logger.info(
        "%s/%s: %d queries (%s)", name, partition, len(ids), "cache hit" if cached else "encoded"
    )
    return {k: v for k, v in record.items() if k != "query_ids"} | {
        "index_path": str(path.resolve()),
        "cache_hit": cached,
        "status": "complete",
    }


def main() -> None:
    import json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/E3_missing_queries.toml")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--prepared", help="prepared input JSON from --prepare-only")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    cfg = load_config(REPO_ROOT / args.config)
    dense_cfg = resolve_dtype(load_config(REPO_ROOT / cfg["dense_config"]), cfg["dtype"])
    dense_cfg["batch_size"] = cfg["batch_size"]
    manifest = load_manifest(REPO_ROOT / cfg["split_manifest"])
    artifacts = (
        Path(os.environ.get("SCRATCH_ROOT", OUTPUTS))
        / "training_queries"
        / manifest["sha256"]
        / cfg["run_name"]
    )
    if args.prepared:
        prepared_path = Path(args.prepared)
        prepared = json.loads(prepared_path.read_text())
    else:
        prepared = prepare_inputs(cfg, manifest, dense_cfg)
        prepared_path = artifacts / f"prepared-{prepared['sha256']}.json"
        write_json(prepared_path, prepared)
    validate_prepared(prepared, cfg, manifest, dense_cfg)
    logger.info(
        "Prepared inputs: %s (%d queries)",
        prepared_path,
        sum(len(p["query_ids"]) for p in prepared["parts"]),
    )
    if args.prepare_only:
        print(prepared_path)
        return

    import torch

    from csrv2_repro.e1_dense_backbone import build_dense_encoder

    if not torch.cuda.is_available():
        raise RuntimeError("Run extraction in a GPU allocation; use --prepare-only on login nodes")
    encoder = build_dense_encoder(dense_cfg)
    cfg = {**cfg, "resolved_dense_config": dense_cfg}
    summary = {
        "exp_id": "E3",
        "run_name": cfg["run_name"],
        "manifest_sha256": manifest["sha256"],
        "prepared_path": str(prepared_path.resolve()),
        "prepared_sha256": prepared["sha256"],
        "source_code_sha256": {
            name: file_sha256(Path(__file__).parent / name)
            for name in (
                "e3_missing_queries.py",
                "e1_dense_backbone.py",
                "encoders.py",
                "splits.py",
            )
        },
        "provenance": provenance(cfg),
        "partitions": [],
        "status": "running",
    }
    output = RESULTS / "E3" / cfg["run_name"] / "summary.json"
    write_json(output, summary)
    for part in prepared["parts"]:
        summary["partitions"].append(
            encode_partition(encoder, part, artifacts, manifest["sha256"], prepared["sha256"])
        )
        write_json(output, summary)
    summary["n_queries"] = sum(p["n_queries"] for p in summary["partitions"])
    summary["status"] = "complete"
    write_json(output, summary)


if __name__ == "__main__":
    main()
