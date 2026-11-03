"""Per-query paired comparison of two retrieval runs on the same queries (used by E1p, ledger D-005).

Per-query nDCG@10 is recomputed with pytrec_eval (the same library and call mteb uses) from the
`<Task>_default_predictions.json` files that mteb writes with save_predictions=True, so the per-query mean
equals mteb's reported score. The decision statistic is an equivalence test (TOST): the runs are declared
equivalent at margin delta if the 90% CI of the mean paired difference lies inside [-delta, +delta]
(= two one-sided tests at alpha=0.05). Non-significance of a plain paired test is not evidence of equivalence.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

NDCG = "ndcg_cut_10"


def load_qrels(task_name: str, split: str = "test") -> tuple[dict, bool]:
    """qrels as mteb loads them, and whether the task ignores query==doc ids (e.g. ArguAna)."""
    import mteb

    task = mteb.get_tasks(tasks=[task_name], languages=["eng"])[0]
    task.load_data()
    qrels = task.relevant_docs[split]
    if "default" in qrels and isinstance(next(iter(qrels["default"].values()), None), dict):
        qrels = qrels["default"]  # mteb keeps hf_subsets as an outer level for some tasks
    return qrels, bool(getattr(task, "ignore_identical_ids", False))


def per_query_ndcg10(predictions_file: Path, qrels: dict, ignore_identical_ids: bool) -> dict[str, float]:
    import pytrec_eval

    with open(predictions_file) as f:
        run = json.load(f)
    if ignore_identical_ids:
        run = {q: {d: s for d, s in docs.items() if d != q} for q, docs in run.items()}
    scores = pytrec_eval.RelevanceEvaluator(qrels, {"ndcg_cut.10"}).evaluate(run)
    return {q: 100 * v[NDCG] for q, v in scores.items()}


def find_predictions(mteb_dir: Path, task_name: str) -> Path:
    hits = sorted(Path(mteb_dir).rglob(f"{task_name}_*predictions.json"))
    if len(hits) != 1:
        raise FileNotFoundError(f"expected 1 predictions file for {task_name} under {mteb_dir}, got {hits}")
    return hits[0]


def paired_stats(a: dict[str, float], b: dict[str, float], delta: float, n_boot: int = 10000,
                 seed: int = 0) -> dict:
    """b - a over the shared queries: mean, bootstrap CIs, sign-flip p-value, TOST verdict at delta."""
    qids = sorted(set(a) & set(b))
    if set(a) != set(b):
        raise ValueError(f"query sets differ: {len(set(a) ^ set(b))} not shared")
    d = np.array([b[q] - a[q] for q in qids])
    rng = np.random.default_rng(seed)
    boot = d[rng.integers(0, len(d), size=(n_boot, len(d)))].mean(axis=1)
    flips = rng.choice([-1.0, 1.0], size=(n_boot, len(d)))
    null = np.abs((flips * d).mean(axis=1))
    ci90 = np.percentile(boot, [5, 95])
    return {
        "n_queries": len(d),
        "mean_a": round(float(np.mean([a[q] for q in qids])), 4),
        "mean_b": round(float(np.mean([b[q] for q in qids])), 4),
        "mean_diff": round(float(d.mean()), 4),
        "ci95": [round(float(x), 4) for x in np.percentile(boot, [2.5, 97.5])],
        "ci90": [round(float(x), 4) for x in ci90],
        "frac_queries_changed": round(float((np.abs(d) > 1e-9).mean()), 4),
        "max_abs_query_diff": round(float(np.abs(d).max()), 4),
        "p_signflip_two_sided": round(float((np.sum(null >= abs(d.mean())) + 1) / (n_boot + 1)), 4),
        "delta": delta,
        "equivalent_tost_0.05": bool(ci90[0] > -delta and ci90[1] < delta),
    }


def macro_stats(per_task: dict[str, tuple[dict, dict]], delta: float, n_boot: int = 10000,
                seed: int = 0) -> dict:
    """Mean over tasks of the per-task mean difference (the paper reports task means), bootstrapping
    queries within each task independently."""
    rng = np.random.default_rng(seed)
    means, boots = [], []
    for a, b in per_task.values():
        d = np.array([b[q] - a[q] for q in sorted(a)])
        means.append(d.mean())
        boots.append(d[rng.integers(0, len(d), size=(n_boot, len(d)))].mean(axis=1))
    boot = np.mean(boots, axis=0)
    ci90 = np.percentile(boot, [5, 95])
    return {
        "tasks": list(per_task),
        "mean_diff": round(float(np.mean(means)), 4),
        "ci95": [round(float(x), 4) for x in np.percentile(boot, [2.5, 97.5])],
        "ci90": [round(float(x), 4) for x in ci90],
        "delta": delta,
        "equivalent_tost_0.05": bool(ci90[0] > -delta and ci90[1] < delta),
    }
