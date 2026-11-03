"""E1: dense Qwen3-Embedding-4B on the 8 retrieval tasks (Table 3, backbone row) + embedding cache.

    uv run python -m csrv2_repro.e1_dense_backbone --config configs/E1_dense_both.toml --dtype bfloat16

The cache written to <cache_root>/<run_name>/ is reused by E2 (MRL truncation) and the CSR evals.
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import torch

from csrv2_repro.common import (
    OUTPUTS,
    RESULTS,
    RETRIEVAL_TASKS,
    load_config,
    provenance,
    resolve_dtype,
    run_mteb,
    upstream_prompts,
    write_json,
)
from csrv2_repro.encoders import CachedEncoder


def cache_root() -> Path:
    return Path(os.environ.get("SCRATCH_ROOT", OUTPUTS)) / "emb_cache"


def build_dense_encoder(cfg: dict) -> CachedEncoder:
    """SentenceTransformer + CachedEncoder for a resolved E1 config (cfg["dtype"] set).

    E2 calls this with the same E1 config, so it hits the same cache (same dir and model_tag).
    """
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(
        cfg["model"],
        revision=cfg.get("revision"),
        device="cuda" if torch.cuda.is_available() else None,
        model_kwargs={"torch_dtype": getattr(torch, cfg["dtype"])},
        tokenizer_kwargs={"padding_side": cfg.get("padding_side", "left")},  # upstream: left
    )
    model.max_seq_length = cfg.get("max_seq_length", 512)
    return CachedEncoder(
        model,
        model_tag="|".join(
            str(x)
            for x in (
                cfg["model"],
                cfg.get("revision"),
                cfg["dtype"],
                model.max_seq_length,
                cfg.get("padding_side", "left"),
            )
        ),
        prompts=upstream_prompts(),
        prompt_mode=cfg["prompt_mode"],
        cache_dir=cache_root() / cfg["run_name"],
        batch_size=cfg.get("batch_size", 16),
        meta_name=f"csrv2-repro/{cfg['run_name']}",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--dtype", help="float32|bfloat16, for configs without dtype (D-005)")
    parser.add_argument("--tasks", nargs="+", help="override the task list (debugging)")
    parser.add_argument(
        "--overwrite", action="store_true", help="recompute mteb scores (cache is kept)"
    )
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    cfg = resolve_dtype(load_config(args.config), args.dtype)
    tasks = args.tasks or cfg.get("tasks", RETRIEVAL_TASKS)
    out_dir = RESULTS / cfg["exp_id"] / cfg["run_name"]
    encoder = build_dense_encoder(cfg)

    res = run_mteb(
        encoder, tasks, out_dir / "mteb", args.overwrite, {"batch_size": encoder.batch_size}
    )
    summary = {
        "exp_id": cfg["exp_id"],
        "run_name": cfg["run_name"],
        "metric": "nDCG@10 x100, test split",
        **res,
        "paper_target_mean": cfg.get("paper_target_mean"),
        "embedding_cache": str(encoder.cache_dir),
        "provenance": provenance(cfg),
    }
    write_json(out_dir / "summary.json", summary)
    logging.info("mean nDCG@10 = %s (paper %s)", summary["mean"], summary["paper_target_mean"])


if __name__ == "__main__":
    main()
