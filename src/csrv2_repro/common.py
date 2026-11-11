"""Shared helpers: paths, task list, upstream prompts, config loading, provenance, result collection."""

from __future__ import annotations

import json
import os
import platform
import socket
import subprocess
import tomllib
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
UPSTREAM_TEXT = REPO_ROOT / "third_party" / "CSRv2" / "text"
OUTPUTS = REPO_ROOT / "outputs"
RESULTS = REPO_ROOT / "results"

# The 8 retrieval tasks of the paper (App. A), as MTEB task names.
RETRIEVAL_TASKS = [
    "ArguAna",
    "NFCorpus",
    "CQADupstackGamingRetrieval",
    "CQADupstackUnixRetrieval",
    "ClimateFEVERHardNegatives",
    "FiQA2018",
    "SCIDOCS",
    "SciFact",
]

# Table 3's six task types (paper App. A; upstream text/task_list.json mapped through dataset_to_task.json).
# STS17 / STS22.v2 are absent from dataset_to_task.json; their MTEB names come from MTEB_task_prompts.json.
TASK_TYPES: dict[str, list[str]] = {
    "classification": [
        "AmazonCounterfactualClassification",
        "EmotionClassification.v2",
        "Banking77Classification.v2",
        "MTOPIntentClassification",
        "MTOPDomainClassification",
        "MassiveIntentClassification",
        "MassiveScenarioClassification",
        "ImdbClassification.v2",
        "ToxicConversationsClassification.v2",
        "TweetSentimentExtractionClassification.v2",
    ],
    "clustering": [
        "ArXivHierarchicalClusteringS2S",
        "BiorxivClusteringP2P.v2",
        "BiorxivClusteringS2S.v2",
        "TwentyNewsgroupsClustering.v2",
        "MedrxivClusteringP2P.v2",
        "MedrxivClusteringS2S.v2",
        "StackExchangeClustering.v2",
        "StackExchangeClusteringP2P.v2",
    ],
    "retrieval": RETRIEVAL_TASKS,
    "sts": [
        "SICK-R",
        "STS12",
        "STS13",
        "STS14",
        "STS15",
        "STS16",
        "STSBenchmark",
        "BIOSSES",
        "STS17",
        "STS22.v2",
    ],
    "reranking": ["StackOverflowDupQuestions", "AskUbuntuDupQuestions", "SciDocsRR"],
    "pair_classification": ["TwitterURLCorpus", "SprintDuplicateQuestions"],
}

# English subsets of multilingual tasks (ledger D-008). Upstream asks mteb for hf_subsets=["en"]; STS17 has no
# "en" subset (that filter would leave it empty), so we take its monolingual "en-en" pair.
HF_SUBSETS: dict[str, list[str]] = {
    "AmazonCounterfactualClassification": ["en"],
    "MTOPIntentClassification": ["en"],
    "MTOPDomainClassification": ["en"],
    "MassiveIntentClassification": ["en"],
    "MassiveScenarioClassification": ["en"],
    "STS17": ["en-en"],
    "STS22.v2": ["en"],
}

# Metric per task type, as in Table 3's header (mteb score key, label). Pair classification: paper App. B says
# cosine-similarity AP, so cosine_ap rather than mteb's main_score (max_ap over four similarities).
TASK_METRICS: dict[str, tuple[str, str]] = {
    "classification": ("accuracy", "accuracy x100"),
    "clustering": ("v_measure", "V-measure x100"),
    "retrieval": ("ndcg_at_10", "nDCG@10 x100"),
    "sts": ("cosine_spearman", "cosine Spearman x100"),
    "pair_classification": ("cosine_ap", "cosine AP x100"),
    "reranking": ("map", "MAP x100"),
}


def resolve_task_type(cfg: dict, override: str | None) -> dict:
    """Set cfg["task_type"] (default retrieval) from the config or --task-type.

    Configs without task_type cover any type: --task-type picks one and suffixes run_name with it, so results
    and embedding caches of different types never mix. A config that pins task_type cannot be overridden.
    """
    cfg = dict(cfg)
    if "task_type" in cfg:
        if override and override != cfg["task_type"]:
            raise ValueError(f"config pins task_type={cfg['task_type']}, refusing --task-type {override}")
        return cfg
    if not override:
        cfg["task_type"] = "retrieval"
        return cfg
    if override not in TASK_TYPES:
        raise ValueError(f"--task-type must be one of {list(TASK_TYPES)}")
    cfg["task_type"] = override
    cfg["run_name"] = f"{cfg['run_name']}_{override}"
    return cfg


def type_value(table, task_type: str):
    """A config value that is either one value (retrieval-era configs) or a table keyed by task type."""
    if isinstance(table, dict) and task_type in table:
        return table[task_type]
    if isinstance(table, dict) and set(table) & set(TASK_TYPES):
        return None
    return table


def upstream_prompts() -> dict[str, str]:
    """Instruction prompts keyed by MTEB task name, exactly as upstream's MTEB evaluation uses them."""
    with open(UPSTREAM_TEXT / "MTEB_task_prompts.json") as f:
        return json.load(f)


def load_config(path: str | Path) -> dict:
    with open(path, "rb") as f:
        return tomllib.load(f)


def _git(*args: str, cwd: Path = REPO_ROOT) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=cwd, text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def provenance(config: dict) -> dict:
    """Everything needed to trace a number in results/ back to code, data and hardware."""
    import torch

    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain", "--", ".", ":!ledger")),
        "upstream_commit": _git("rev-parse", "HEAD", cwd=UPSTREAM_TEXT),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "host": socket.gethostname(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "python": platform.python_version(),
        "packages": {
            p: version(p)
            for p in ["torch", "transformers", "sentence-transformers", "mteb", "peft"]
        },
        "config": config,
    }


def collect_scores(mteb_output_dir: Path, tasks: list[str], metric: str) -> dict[str, float | None]:
    """Read the test-split `metric` (x100, as in the paper) per task from mteb result JSONs under a folder.

    Multi-subset tasks average over the subsets that ran (one English subset each here, see HF_SUBSETS).
    """
    scores: dict[str, float | None] = {}
    for task in tasks:
        hits = sorted(mteb_output_dir.rglob(f"{task}.json"))
        if not hits:
            scores[task] = None
            continue
        with open(hits[-1]) as f:
            test = json.load(f)["scores"]["test"]
        scores[task] = round(100 * sum(s[metric] for s in test) / len(test), 2)
    return scores


def collect_ndcg10(mteb_output_dir: Path, tasks: list[str]) -> dict[str, float | None]:
    return collect_scores(mteb_output_dir, tasks, "ndcg_at_10")


DTYPE_TAGS = {"float32": "fp32", "bfloat16": "bf16"}


def resolve_dtype(cfg: dict, override: str | None) -> dict:
    """Set cfg["dtype"] from the config or the --dtype CLI flag (ledger D-005).

    Configs that leave dtype out (the 8-task runs, pending D-005) must get it from --dtype, and their
    run_name gets a _fp32/_bf16 suffix so results and embedding caches of different dtypes never mix.
    A config that pins dtype cannot be overridden (its run_name already names the dtype).
    """
    cfg = dict(cfg)
    if "dtype" in cfg:
        if override and override != cfg["dtype"]:
            raise ValueError(f"config pins dtype={cfg['dtype']}, refusing --dtype {override}")
        return cfg
    if not override:
        raise ValueError("this config has no dtype (pending ledger D-005): pass --dtype float32|bfloat16")
    if override not in DTYPE_TAGS:
        raise ValueError(f"--dtype must be one of {list(DTYPE_TAGS)}")
    cfg["dtype"] = override
    cfg["run_name"] = f"{cfg['run_name']}_{DTYPE_TAGS[override]}"
    return cfg


def run_mteb(
    model, tasks: list[str], out: Path, overwrite: bool, encode_kwargs: dict, task_type: str = "retrieval"
) -> dict:
    """mteb.MTEB.run on the test split; returns per-task scores in the task type's Table 3 metric.

    Retrieval also saves per-query rankings (top 100) for paired tests; top_k=100 truncates the run before
    mteb scores it, so metrics @<=100 are exact, @1000 are not. Finished tasks are skipped on rerun
    (resumable jobs) unless overwrite.
    """
    import shutil

    import mteb

    if overwrite and out.exists():
        shutil.rmtree(out)
    task_objs = [mteb.get_task(t, languages=["eng"], hf_subsets=HF_SUBSETS.get(t)) for t in tasks]
    extra = {"save_predictions": True, "top_k": 100} if task_type == "retrieval" else {}
    mteb.MTEB(tasks=task_objs).run(
        model,
        eval_splits=["test"],
        output_folder=str(out),
        show_progress_bar=True,
        encode_kwargs=encode_kwargs,
        **extra,
    )
    scores = collect_scores(out, tasks, TASK_METRICS[task_type][0])
    return {"per_task": scores, "mean": mean_or_none(scores.values())}


def mean_or_none(values) -> float | None:
    values = list(values)
    if not values or any(v is None for v in values):
        return None
    return round(sum(values) / len(values), 2)


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
        f.write("\n")
