"""E1: dense Qwen3-Embedding-4B on the 8 retrieval tasks (Table 3, backbone row) + embedding cache.

    uv run python -m csrv2_repro.e1_dense_backbone --config configs/E1_dense_both.toml

The cache written to <cache_root>/<run_name>/ is reused by E2 (MRL truncation) and the CSR evals.
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
from pathlib import Path

import torch

from csrv2_repro.common import (
    OUTPUTS,
    RESULTS,
    RETRIEVAL_TASKS,
    collect_ndcg10,
    load_config,
    mean_or_none,
    provenance,
    upstream_prompts,
    write_json,
)
from csrv2_repro.encoders import CachedEncoder


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--tasks", nargs="+", help="override the task list (debugging)")
    parser.add_argument(
        "--overwrite", action="store_true", help="recompute mteb scores (cache is kept)"
    )
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    cfg = load_config(args.config)
    tasks = args.tasks or cfg.get("tasks", RETRIEVAL_TASKS)
    run_name = cfg["run_name"]
    cache_root = Path(os.environ.get("SCRATCH_ROOT", OUTPUTS)) / "emb_cache"
    out_dir = RESULTS / cfg["exp_id"] / run_name
    mteb_dir = out_dir / "mteb"
    if args.overwrite and mteb_dir.exists():
        shutil.rmtree(mteb_dir)

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(
        cfg["model"],
        revision=cfg.get("revision"),
        device="cuda" if torch.cuda.is_available() else None,
        model_kwargs={"torch_dtype": getattr(torch, cfg.get("dtype", "float32"))},
        tokenizer_kwargs={"padding_side": cfg.get("padding_side", "left")},  # upstream: left
    )
    model.max_seq_length = cfg.get("max_seq_length", 512)
    encoder = CachedEncoder(
        model,
        model_tag="|".join(
            str(x)
            for x in (
                cfg["model"],
                cfg.get("revision"),
                cfg.get("dtype", "float32"),
                model.max_seq_length,
                cfg.get("padding_side", "left"),
            )
        ),
        prompts=upstream_prompts(),
        prompt_mode=cfg["prompt_mode"],
        cache_dir=cache_root / run_name,
        batch_size=cfg.get("batch_size", 16),
        meta_name=f"csrv2-repro/{run_name}",
    )

    import mteb

    evaluation = mteb.MTEB(tasks=mteb.get_tasks(tasks=tasks, languages=["eng"]))
    evaluation.run(
        encoder,
        eval_splits=["test"],
        output_folder=str(mteb_dir),
        encode_kwargs={"batch_size": encoder.batch_size},
        overwrite_results=args.overwrite,
    )

    per_task = collect_ndcg10(mteb_dir, tasks)
    summary = {
        "exp_id": cfg["exp_id"],
        "run_name": run_name,
        "metric": "nDCG@10 x100, test split",
        "per_task": per_task,
        "mean": mean_or_none(per_task.values()),
        "paper_target_mean": cfg.get("paper_target_mean"),
        "embedding_cache": str(cache_root / run_name),
        "provenance": provenance(cfg),
    }
    write_json(out_dir / "summary.json", summary)
    logging.info("mean nDCG@10 = %s (paper %s)", summary["mean"], summary["paper_target_mean"])


if __name__ == "__main__":
    main()
