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


def collect_ndcg10(mteb_output_dir: Path, tasks: list[str]) -> dict[str, float | None]:
    """Read test nDCG@10 (x100, as in the paper) per task from mteb result JSONs under a folder."""
    scores: dict[str, float | None] = {}
    for task in tasks:
        hits = sorted(mteb_output_dir.rglob(f"{task}.json"))
        if not hits:
            scores[task] = None
            continue
        with open(hits[-1]) as f:
            test = json.load(f)["scores"]["test"]
        scores[task] = round(100 * sum(s["ndcg_at_10"] for s in test) / len(test), 2)
    return scores


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


def run_mteb(model, tasks: list[str], out: Path, overwrite: bool, encode_kwargs: dict) -> dict:
    """mteb.MTEB.run on the test split, saving per-query rankings (top 100) for paired tests.

    top_k=100 truncates the run before mteb scores it: metrics @<=100 are exact, @1000 are not.
    Finished tasks are skipped on rerun (resumable jobs) unless overwrite.
    """
    import shutil

    import mteb

    if overwrite and out.exists():
        shutil.rmtree(out)
    mteb.MTEB(tasks=mteb.get_tasks(tasks=tasks, languages=["eng"])).run(
        model,
        eval_splits=["test"],
        output_folder=str(out),
        show_progress_bar=True,
        encode_kwargs=encode_kwargs,
        save_predictions=True,
        top_k=100,
    )
    scores = collect_ndcg10(out, tasks)
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
