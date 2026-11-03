"""E1p: bf16 vs fp32 inference report (ledger D-005).

Reads two pairs of finished runs (each pair = same config except dtype) and reports:
  - nDCG@10 per task and the bf16 - fp32 delta;
  - dense agreement: cosine between the fp32 and bf16 cached embeddings of the same texts;
  - sparse agreement (released CSRv2 ckpt): for each k, Jaccard overlap of the TopK active latent sets
    and the fraction of texts whose active set is identical, SAE applied in fp32 to both caches.

    uv run python -m csrv2_repro.precision_report \
        --dense results/E1p/qwen3_4b_dense_fp32 results/E1p/qwen3_4b_dense_bf16 \
        --sparse results/E0/released_ckpt_fp32 results/E0/released_ckpt_bf16
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from csrv2_repro.common import RESULTS, provenance, write_json


def load_summary(run_dir: Path) -> dict:
    with open(Path(run_dir) / "summary.json") as f:
        return json.load(f)


def paired_cache_files(cache_a: Path, cache_b: Path):
    """Yield (task, side, emb_a, emb_b) for cache files covering the same texts.

    Filenames are <side>-<n>-<hash>; the hash includes the dtype, so pair on (task, side, n).
    """
    for task_dir in sorted(Path(cache_a).iterdir()):
        for fa in sorted(task_dir.glob("*.npy")):
            side, n, _ = fa.stem.split("-")
            fb = list((Path(cache_b) / task_dir.name).glob(f"{side}-{n}-*.npy"))
            if len(fb) == 1:
                yield task_dir.name, side, np.load(fa), np.load(fb[0])


def ndcg_table(a: dict, b: dict, key_a=None, key_b=None) -> dict:
    sa = a if key_a is None else a[key_a]
    sb = b if key_b is None else b[key_b]
    rows = {}
    for task, va in sa["per_task"].items():
        vb = sb["per_task"].get(task)
        rows[task] = {
            "fp32": va,
            "bf16": vb,
            "delta": None if None in (va, vb) else round(vb - va, 2),
        }
    rows["mean"] = {
        "fp32": sa["mean"],
        "bf16": sb["mean"],
        "delta": None if None in (sa["mean"], sb["mean"]) else round(sb["mean"] - sa["mean"], 2),
    }
    return rows


def load_released_sae(summary: dict):
    from sentence_transformers.sparse_encoder.models import SparseAutoEncoder

    cfg = summary["provenance"]["config"]
    sae = SparseAutoEncoder.load(
        cfg["model"], subfolder="3_SparseAutoEncoder", revision=cfg.get("revision")
    )
    return sae.eval()


def active_set_agreement(sae, a: np.ndarray, b: np.ndarray, k: int, batch_size: int = 2048) -> dict:
    """Overlap of the TopK supports (k largest pre-activations, before ReLU) of paired embeddings."""
    inter = []
    with torch.inference_mode():
        for i in range(0, len(a), batch_size):
            masks = []
            for x in (a, b):
                pre = sae.encode_pre_act(sae.preprocess(torch.from_numpy(x[i : i + batch_size]))[0])
                idx = torch.topk(pre, k, dim=-1).indices
                masks.append(torch.zeros_like(pre, dtype=torch.bool).scatter_(1, idx, True))
            inter.append((masks[0] & masks[1]).sum(dim=1).float())
    inter = torch.cat(inter)
    return {
        "jaccard": round((inter / (2 * k - inter)).mean().item(), 4),
        "identical_frac": round((inter == k).float().mean().item(), 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--dense", nargs=2, metavar=("FP32_RUN", "BF16_RUN"), required=True)
    parser.add_argument("--sparse", nargs=2, metavar=("FP32_RUN", "BF16_RUN"), required=True)
    parser.add_argument("--out", default=str(RESULTS / "E1p" / "precision_report.json"))
    args = parser.parse_args()

    report: dict = {"dense": {}, "sparse": {}}

    d32, d16 = (load_summary(p) for p in args.dense)
    report["dense"]["ndcg10"] = ndcg_table(d32, d16)
    report["dense"]["cosine_fp32_vs_bf16"] = {
        f"{task}/{side}": {
            "mean": round(float(cos.mean()), 6),
            "min": round(float(cos.min()), 6),
        }
        for task, side, a, b in paired_cache_files(d32["embedding_cache"], d16["embedding_cache"])
        for cos in [(a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))]
    }

    s32, s16 = (load_summary(p) for p in args.sparse)
    report["sparse"]["ndcg10_by_k"] = {
        k: ndcg_table(s32["per_k"][k], s16["per_k"][k]) for k in s32["per_k"] if k in s16["per_k"]
    }
    report["sparse"]["faithful_minus_cached"] = {
        "fp32": s32.get("faithful_minus_cached"),
        "bf16": s16.get("faithful_minus_cached"),
    }
    sae = load_released_sae(s32)
    pairs = list(paired_cache_files(s32["embedding_cache"], s16["embedding_cache"]))
    report["sparse"]["active_set_agreement"] = {}
    for k in map(int, report["sparse"]["ndcg10_by_k"]):
        report["sparse"]["active_set_agreement"][str(k)] = {
            f"{task}/{side}": active_set_agreement(sae, a, b, k) for task, side, a, b in pairs
        }

    report["provenance"] = provenance({"dense": args.dense, "sparse": args.sparse})
    write_json(Path(args.out), report)

    print("dense nDCG@10:", json.dumps(report["dense"]["ndcg10"]))
    for k, t in report["sparse"]["ndcg10_by_k"].items():
        agree = report["sparse"]["active_set_agreement"][k]
        print(
            f"sparse k={k}: nDCG delta mean {t['mean']['delta']} | jaccard",
            {n: v["jaccard"] for n, v in agree.items()},
        )
    print("written", args.out)


if __name__ == "__main__":
    main()
