"""E0: evaluate the released Y-Research-Group/CSRv2-retrieval checkpoint.

Two paths, both with upstream's protocol (left padding, MTEB_task_prompts.json on queries and documents,
plain mteb.MTEB.run):
  faithful - SparseEncoder passed to mteb exactly like eval_all() in
             third_party/CSRv2/text/all_step_pipeline_mteb_evaluation.py; one full backbone pass per k.
  cached   - backbone+LoRA+Normalize embeddings cached once, SAE TopK applied per k (CachedEncoder).
For every k in faithful_ks both paths run, and the summary reports their difference (should be ~0).

    uv run python -m csrv2_repro.e0_released_ckpt --config configs/E0_released_ckpt.toml
    uv run python -m csrv2_repro.e0_released_ckpt --config configs/E6b_released_ckpt.toml --dtype bfloat16
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import torch

from csrv2_repro.common import (
    OUTPUTS,
    RETRIEVAL_TASKS,
    load_config,
    provenance,
    resolve_dtype,
    run_mteb,
    upstream_prompts,
    write_json,
)
from csrv2_repro.encoders import CachedEncoder, sae_topk_transform
from csrv2_repro.splits import evaluation_context


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--dtype", help="float32|bfloat16, for configs without dtype (D-005)")
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
    tasks = args.tasks or cfg.get("tasks", RETRIEVAL_TASKS)
    ks = args.ks or cfg["ks"]
    faithful_ks = [k for k in cfg.get("faithful_ks", []) if k in ks]
    dtype = cfg["dtype"]
    batch_size = cfg.get("batch_size", 2)
    cfg["ks"] = ks
    cfg, manifest, out_dir = evaluation_context(cfg, tasks, args.split_manifest)
    cache_root = Path(os.environ.get("SCRATCH_ROOT", OUTPUTS)) / "emb_cache"

    from sentence_transformers import SentenceTransformer, SparseEncoder
    from sentence_transformers.sparse_encoder.models import SparseAutoEncoder

    model = SparseEncoder(
        cfg["model"],
        revision=cfg.get("revision"),
        device="cuda" if torch.cuda.is_available() else None,
        model_kwargs={"torch_dtype": getattr(torch, dtype)},
        tokenizer_kwargs={"padding_side": "left"},
    )
    model.prompts = upstream_prompts()
    sae_idx, sae = next((i, m) for i, m in enumerate(model) if isinstance(m, SparseAutoEncoder))
    logging.info(
        "loaded %s: SAE %d -> %d, default k=%d", cfg["model"], sae.input_dim, sae.hidden_dim, sae.k
    )

    # Dense part of the same model (shares weights): Transformer(+LoRA) -> Pooling -> Normalize.
    backbone = SentenceTransformer(modules=[model[i] for i in range(sae_idx)], device=model.device)
    encoder = CachedEncoder(
        backbone,
        model_tag="|".join(
            str(x) for x in (cfg["model"], cfg.get("revision"), dtype, model.max_seq_length, "left")
        ),
        prompts=upstream_prompts(),
        prompt_mode="both",
        cache_dir=cache_root / cfg["run_name"],
        batch_size=batch_size,
        meta_name=f"csrv2-repro/{cfg['run_name']}",
    )

    summary = {
        "exp_id": cfg["exp_id"],
        "run_name": cfg["run_name"],
        "metric": "nDCG@10 x100, manifest test queries" if manifest else "nDCG@10 x100, test split",
        "evaluation": cfg.get("evaluation", {"protocol": "full-test"}),
        "per_k": {},
        "faithful_per_k": {},
        "faithful_minus_cached": {},
        "paper_target_mean_by_k": cfg.get("paper_target_mean_by_k"),
        "embedding_cache": str(cache_root / cfg["run_name"]),
    }
    for k in ks:
        encoder.transform = sae_topk_transform(sae, k)
        summary["per_k"][str(k)] = run_mteb(
            encoder, tasks, out_dir / "mteb" / f"k{k}", args.overwrite, {"batch_size": batch_size},
            split_manifest=manifest,
        )
        logging.info("cached k=%d nDCG@10 %s", k, summary["per_k"][str(k)])

        if k in faithful_ks:
            # SparseEncoder copies the SAE's k into max_active_dims at load time and passes it to
            # forward(), overriding sae.k, so both must change (same as editing config.json + reload).
            sae.k = k
            model.max_active_dims = k
            faithful = run_mteb(
                model,
                tasks,
                out_dir / "mteb" / f"faithful_k{k}",
                args.overwrite,
                {"convert_to_sparse_tensor": False, "batch_size": batch_size},
                split_manifest=manifest,
            )
            summary["faithful_per_k"][str(k)] = faithful
            summary["faithful_minus_cached"][str(k)] = {
                t: None
                if faithful["per_task"][t] is None
                or summary["per_k"][str(k)]["per_task"][t] is None
                else round(faithful["per_task"][t] - summary["per_k"][str(k)]["per_task"][t], 2)
                for t in tasks
            }
            logging.info("faithful k=%d nDCG@10 %s", k, faithful)

        # Write after every k so a timed-out job keeps what it finished.
        write_json(out_dir / "summary.json", {**summary, "provenance": provenance(cfg)})


if __name__ == "__main__":
    main()
