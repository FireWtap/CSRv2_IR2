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


def paired_section(mteb_a: Path, mteb_b: Path, ref_a: dict, ref_b: dict, delta: float,
                   qrels_cache: dict) -> tuple[dict, list[dict]]:
    """Per-task and macro paired stats (b - a) from mteb predictions; checks the per-query means
    reproduce mteb's reported nDCG@10. Returns (stats, per-query rows)."""
    from csrv2_repro import paired

    stats, pairs, rows = {}, {}, []
    for task in ref_a["per_task"]:
        if task not in qrels_cache:
            qrels_cache[task] = paired.load_qrels(task)
        qrels, ignore_ids = qrels_cache[task]
        qa = paired.per_query_ndcg10(paired.find_predictions(mteb_a, task), qrels, ignore_ids)
        qb = paired.per_query_ndcg10(paired.find_predictions(mteb_b, task), qrels, ignore_ids)
        for q, ref, name in ((qa, ref_a, "a"), (qb, ref_b, "b")):
            mean = round(float(np.mean(list(q.values()))), 2)
            if abs(mean - ref["per_task"][task]) > 0.011:
                raise RuntimeError(f"{task}/{name}: per-query mean {mean} != mteb {ref['per_task'][task]}")
        stats[task] = paired.paired_stats(qa, qb, delta)
        pairs[task] = (qa, qb)
        rows += [{"task": task, "query_id": q, "fp32": qa[q], "bf16": qb[q]} for q in sorted(qa)]
    stats["macro"] = paired.macro_stats(pairs, delta)
    return stats, rows


def write_rows(path: Path, rows: list[dict]) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: round(v, 4) if isinstance(v, float) else v for k, v in r.items()})


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--dense", nargs=2, metavar=("FP32_RUN", "BF16_RUN"), required=True)
    parser.add_argument("--sparse", nargs=2, metavar=("FP32_RUN", "BF16_RUN"), required=True)
    parser.add_argument("--out", default=str(RESULTS / "E1p" / "precision_report.json"))
    parser.add_argument("--delta", type=float, default=0.2,
                        help="TOST equivalence margin in nDCG@10 points (proposal, see ledger D-005)")
    args = parser.parse_args()

    report: dict = {"dense": {}, "sparse": {}}
    qrels_cache: dict = {}
    out = Path(args.out)

    d32, d16 = (load_summary(p) for p in args.dense)
    report["dense"]["ndcg10"] = ndcg_table(d32, d16)
    report["dense"]["paired_bf16_minus_fp32"], rows = paired_section(
        Path(args.dense[0]) / "mteb", Path(args.dense[1]) / "mteb", d32, d16, args.delta, qrels_cache
    )
    write_rows(out.with_name("per_query_dense.csv"), rows)
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
    report["sparse"]["paired_bf16_minus_fp32"], rows = {}, []
    for k in report["sparse"]["ndcg10_by_k"]:
        report["sparse"]["paired_bf16_minus_fp32"][k], k_rows = paired_section(
            Path(args.sparse[0]) / "mteb" / f"k{k}", Path(args.sparse[1]) / "mteb" / f"k{k}",
            s32["per_k"][k], s16["per_k"][k], args.delta, qrels_cache,
        )
        rows += [{"k": int(k), **r} for r in k_rows]
    write_rows(out.with_name("per_query_sparse.csv"), rows)
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

    report["provenance"] = provenance({"dense": args.dense, "sparse": args.sparse, "delta": args.delta})
    write_json(out, report)

    print("dense nDCG@10:", json.dumps(report["dense"]["ndcg10"]))
    m = report["dense"]["paired_bf16_minus_fp32"]["macro"]
    print(f"dense paired macro diff {m['mean_diff']} ci90 {m['ci90']} equivalent@{args.delta}: "
          f"{m['equivalent_tost_0.05']}")
    for k, s in report["sparse"]["paired_bf16_minus_fp32"].items():
        m = s["macro"]
        print(f"sparse k={k} paired macro diff {m['mean_diff']} ci90 {m['ci90']} equivalent@{args.delta}: "
              f"{m['equivalent_tost_0.05']}")
    for k, t in report["sparse"]["ndcg10_by_k"].items():
        agree = report["sparse"]["active_set_agreement"][k]
        print(
            f"sparse k={k}: nDCG delta mean {t['mean']['delta']} | jaccard",
            {n: v["jaccard"] for n, v in agree.items()},
        )
    print("written", args.out)


if __name__ == "__main__":
    main()
