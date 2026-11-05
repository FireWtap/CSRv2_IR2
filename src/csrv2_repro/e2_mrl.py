"""E2: MRL baseline (Table 3, MRL column) = cached E1 dense embeddings truncated to the first k dims.

Qwen3-Embedding-4B is natively MRL-trained, so no finetuning (targets.md). Cosine similarity on the
truncated vectors (mteb default, as in the paper). Reuses the E1 embedding cache of the referenced E1
config and dtype, so run it after E1 with the same --dtype; on a cache miss it encodes from scratch.

    uv run python -m csrv2_repro.e2_mrl --config configs/E2_mrl_both.toml --dtype bfloat16
"""

from __future__ import annotations

import argparse
import logging

import numpy as np

from csrv2_repro.common import (
    REPO_ROOT,
    RETRIEVAL_TASKS,
    load_config,
    provenance,
    resolve_dtype,
    run_mteb,
    write_json,
)
from csrv2_repro.e1_dense_backbone import build_dense_encoder
from csrv2_repro.splits import evaluation_context


def truncate(k: int):
    def transform(x: np.ndarray) -> np.ndarray:
        return np.ascontiguousarray(x[:, :k])

    return transform


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--dtype", help="float32|bfloat16, must match the E1 run (D-005)")
    parser.add_argument("--tasks", nargs="+", help="override the task list (debugging)")
    parser.add_argument("--ks", nargs="+", type=int, help="override the k list (debugging)")
    parser.add_argument("--split-manifest", help="evaluate the saved held-out query partitions")
    parser.add_argument(
        "--overwrite", action="store_true", help="recompute mteb scores (cache is kept)"
    )
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    cfg = resolve_dtype(load_config(args.config), args.dtype)
    dense_cfg = resolve_dtype(load_config(REPO_ROOT / cfg["dense_config"]), args.dtype)
    if dense_cfg["dtype"] != cfg["dtype"]:
        raise ValueError(f"E2 dtype {cfg['dtype']} != E1 dtype {dense_cfg['dtype']}")
    tasks = args.tasks or cfg.get("tasks", dense_cfg.get("tasks", RETRIEVAL_TASKS))
    ks = args.ks or cfg["ks"]
    cfg["ks"] = ks
    cfg, manifest, out_dir = evaluation_context(cfg, tasks, args.split_manifest)

    encoder = build_dense_encoder(dense_cfg)
    dim = encoder.model.get_sentence_embedding_dimension()
    if max(ks) > dim:
        raise ValueError(f"k={max(ks)} > embedding dim {dim}")

    summary = {
        "exp_id": cfg["exp_id"],
        "run_name": cfg["run_name"],
        "metric": "nDCG@10 x100, manifest test queries" if manifest else "nDCG@10 x100, test split",
        "evaluation": cfg.get("evaluation", {"protocol": "full-test"}),
        "dense_run": dense_cfg["run_name"],
        "per_k": {},
        "paper_target_mean_by_k": cfg.get("paper_target_mean_by_k"),
        "embedding_cache": str(encoder.cache_dir),
    }
    for k in ks:
        encoder.transform = truncate(k)
        # Distinct mteb model name per k, so mteb never mistakes one k's results for another's.
        encoder.mteb_model_meta = encoder.mteb_model_meta.model_copy(
            update={"name": f"csrv2-repro/{cfg['run_name']}-k{k}"}
        )
        summary["per_k"][str(k)] = run_mteb(
            encoder, tasks, out_dir / "mteb" / f"k{k}", args.overwrite,
            {"batch_size": encoder.batch_size},
            split_manifest=manifest,
        )
        logging.info("MRL k=%d nDCG@10 %s", k, summary["per_k"][str(k)])
        write_json(out_dir / "summary.json", {**summary, "provenance": provenance(cfg)})


if __name__ == "__main__":
    main()
