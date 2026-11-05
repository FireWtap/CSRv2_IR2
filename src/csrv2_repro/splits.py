"""Query-disjoint retrieval splits shared by training and evaluation (D-007).

Only qrels are needed to build the manifest; corpora remain intact. Never split qrel rows.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np

PROTOCOL = "query-heldout-v1"
PARTITIONS = ("train", "validation", "test")


def digest(value) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def qrels_from_rows(rows) -> dict[str, dict[str, int]]:
    """Preserve all judgments, including zero/graded relevance and shared documents."""
    qrels = {}
    for row in rows:
        qid, did = str(row["query-id"]), str(row["corpus-id"])
        score = row["score"]
        if (
            not qid
            or not did
            or not math.isfinite(float(score))
            or int(score) != score
            or score < 0
        ):
            raise ValueError(f"Invalid relevance judgment: {row}")
        docs = qrels.setdefault(qid, {})
        if did in docs and docs[did] != int(score):
            raise ValueError(f"Conflicting judgments for {qid}/{did}")
        docs[did] = int(score)
    return qrels


def partition_queries(ids, fraction: float, seed: int) -> tuple[list[str], list[str]]:
    """Sort before a local seeded shuffle: independent of row order and Python hash seed."""
    ids = sorted(ids)
    if not 0 <= fraction < 1:
        raise ValueError("Split fraction must be in [0, 1)")
    if fraction == 0:
        return ids, []
    size = math.ceil(len(ids) * fraction)
    if not ids or size >= len(ids):
        raise ValueError("Too few queries to produce nonempty training and held-out partitions")
    order = np.random.RandomState(seed).permutation(len(ids))
    return sorted(ids[i] for i in order[size:]), sorted(ids[i] for i in order[:size])


def build_task_split(
    qrels: dict, source: dict, seed=42, test_fraction=0.2, validation_fraction=0.0
) -> dict:
    if not 0 < test_fraction < 1 or not 0 <= validation_fraction < 1:
        raise ValueError("Invalid test/validation fraction")
    if not qrels.get("test"):
        raise ValueError("A nonempty source test split is required")
    aliases = [s for s in ("dev", "validation", "val") if s in qrels]
    if len(aliases) > 1:
        raise ValueError("Multiple validation split aliases; resolve the source explicitly")
    if set(qrels) - {"train", "test", *aliases}:
        raise ValueError(f"Unexpected qrel splits: {sorted(qrels)}")
    seen = set()
    for split, values in qrels.items():
        if not values:
            raise ValueError(f"Empty official split: {split}")
        overlap = seen & set(values)
        if overlap:
            raise ValueError(
                f"Source query IDs overlap across official splits: {sorted(overlap)[:5]}"
            )
        seen.update(values)

    if "train" in qrels:
        train_source = "train"
        train, test = sorted(qrels["train"]), sorted(qrels["test"])
        strategy = "official"
    else:
        if aliases:
            raise ValueError("No train split but validation exists; requires an explicit policy")
        train_source = "test"
        train, test = partition_queries(qrels["test"], test_fraction, seed)
        strategy = "fallback"
    validation_source = aliases[0] if aliases else train_source
    if aliases:
        validation = sorted(qrels[validation_source])
    else:
        train, validation = partition_queries(train, validation_fraction, seed)

    assignments = {"train": train, "validation": validation, "test": test}
    sources = {"train": train_source, "validation": validation_source, "test": "test"}
    entry = {
        "source": source,
        "strategy": strategy,
        "source_qrels_sha256": {s: digest(q) for s, q in qrels.items()},
        "partitions": {
            part: {
                "source_split": sources[part],
                "query_ids": ids,
                "n_queries": len(ids),
                "n_judgments": sum(len(qrels[sources[part]][q]) for q in ids),
            }
            for part, ids in assignments.items()
        },
    }
    validate_entry(entry)
    return entry


def validate_entry(entry: dict) -> None:
    if not entry["source"].get("path") or not entry["source"].get("revision"):
        raise ValueError("Dataset path and pinned revision are required")
    seen = set()
    if set(entry["partitions"]) != set(PARTITIONS):
        raise ValueError("Manifest must contain train, validation and test partitions")
    for part in PARTITIONS:
        spec = entry["partitions"][part]
        ids = spec["query_ids"]
        if ids != sorted(set(ids)) or any(not isinstance(q, str) for q in ids):
            raise ValueError(f"{part}: query IDs must be unique sorted strings")
        if seen & set(ids):
            raise ValueError("Train/validation/test query overlap in manifest")
        seen.update(ids)
        if spec["n_queries"] != len(ids) or (part != "validation" and not ids):
            raise ValueError(f"Invalid query count for {part}")
        if spec["source_split"] not in entry["source_qrels_sha256"]:
            raise ValueError(f"Unknown source split for {part}")


def seal_manifest(tasks: dict, seed=42, test_fraction=0.2, validation_fraction=0.0) -> dict:
    manifest = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "decision": "D-007",
        "seed": seed,
        "test_fraction": test_fraction,
        "validation_fraction": validation_fraction,
        "algorithm": "sorted-query-ids-numpy-RandomState-permutation-ceil-v1",
        "tasks": tasks,
    }
    manifest["sha256"] = digest(manifest)
    validate_manifest(manifest)
    return manifest


def validate_manifest(manifest: dict) -> None:
    if manifest.get("schema_version") != 1 or manifest.get("protocol") != PROTOCOL:
        raise ValueError("Unsupported split manifest")
    if not manifest.get("tasks"):
        raise ValueError("Empty split manifest")
    if manifest.get("sha256") != digest({k: v for k, v in manifest.items() if k != "sha256"}):
        raise ValueError("Split manifest checksum mismatch")
    for entry in manifest["tasks"].values():
        validate_entry(entry)


def load_manifest(path: str | Path) -> dict:
    manifest = json.loads(Path(path).read_text())
    validate_manifest(manifest)
    return manifest


def select_qrels(entry: dict, source_qrels: dict, partition: str) -> dict:
    """Training AND evaluation must select through the same saved manifest entry."""
    validate_entry(entry)
    spec = entry["partitions"][partition]
    source = spec["source_split"]
    qrels = source_qrels[source]
    if digest(qrels) != entry["source_qrels_sha256"][source]:
        raise ValueError(f"Qrels changed for source split {source}")
    if not set(spec["query_ids"]) <= set(qrels):
        raise ValueError("Manifest references missing source queries")
    selected = {q: qrels[q] for q in spec["query_ids"]}
    if sum(map(len, selected.values())) != spec["n_judgments"]:
        raise ValueError("Manifest judgment count mismatch")
    return selected


def load_source(source: dict, config: str = "default"):
    """Load pinned HF data into the project's artifact cache, never an unpinned MTEB loader."""
    from datasets import load_dataset

    from csrv2_repro.common import OUTPUTS

    cache = Path(os.environ.get("SCRATCH_ROOT", OUTPUTS)) / "hf" / "datasets"
    try:
        data = load_dataset(
            source["path"], config, revision=source["revision"], cache_dir=str(cache)
        )
    except ValueError as exc:
        # The pinned SciFact snapshot predates HF config metadata. Its corpus/queries JSONL
        # files exist, but datasets only discovers qrels as the default config.
        if config not in {"queries", "corpus"} or f"BuilderConfig '{config}' not found" not in str(
            exc
        ):
            raise
        from huggingface_hub import hf_hub_download

        path = Path(
            hf_hub_download(
                source["path"],
                f"{config}.jsonl",
                repo_type="dataset",
                revision=source["revision"],
            )
        )
        if path.parent.name != source["revision"] or path.parent.parent.name != "snapshots":
            raise ValueError("Downloaded JSONL does not belong to the pinned snapshot")
        # Here the input file is pinned and verified directly; the JSON builder's cache key
        # is a transform hash, unlike the repository builder's revision directory below.
        return load_dataset("json", data_files={config: str(path)}, cache_dir=str(cache))
    # datasets 3.3.2 may fall back to its *latest* cached revision on a Hub error/offline.
    # Its Arrow cache directory contains the source commit; reject an unverified fallback,
    # including for corpus/query text where the qrel checksum alone would not detect drift.
    for split in data.values():
        if not split.cache_files or any(
            Path(item["filename"]).parent.name != source["revision"] for item in split.cache_files
        ):
            raise ValueError(f"Dataset cache does not match pinned revision: {source}")
    return data


def load_source_qrels(source: dict) -> dict:
    return {split: qrels_from_rows(rows) for split, rows in load_source(source).items()}


def training_qrels(manifest: dict, task_name: str, partition: str = "train") -> dict:
    """Shared input contract for future LoRA and SAE preparation; no test fallback here."""
    validate_manifest(manifest)
    if partition not in ("train", "validation"):
        raise ValueError("Training preparation accepts only train or validation")
    entry = manifest["tasks"][task_name]
    return select_qrels(entry, load_source_qrels(entry["source"]), partition)


def load_pinned_test(task, source: dict) -> None:
    if task.data_loaded:
        return
    qrels = load_source_qrels(source)["test"]
    # Match MTEB 1.38.39's HFDataLoader: take the first corpus/queries split and concatenate
    # title + space + text. Corpus documents are never filtered by query partition.
    corpus = next(iter(load_source(source, "corpus").values()))
    queries = next(iter(load_source(source, "queries").values()))
    task.corpus = {
        "test": {str(row["_id"]): row.get("title", "") + " " + row["text"] for row in corpus}
    }
    task.queries = {
        "test": {str(row["_id"]): row["text"] for row in queries if str(row["_id"]) in qrels}
    }
    task.relevant_docs = {"test": qrels}
    task.data_loaded = True


def configure_mteb_task(task, manifest: dict):
    """Load only when MTEB evaluates the task; completed tasks need no data reload on resume."""
    applied = False

    def load_data(**kwargs):
        nonlocal applied
        if not applied:
            filter_mteb_task(task, manifest)
            applied = True

    task.load_data = load_data
    return task


def filter_mteb_task(task, manifest: dict):
    """Restrict queries/qrels before MTEB scoring; keep the entire corpus unchanged."""
    name = task.metadata.name
    entry = manifest["tasks"][name]  # Missing tasks must never fall back to full test.
    dataset = task.metadata.dataset
    for field in ("path", "revision"):
        if dataset[field] != entry["source"][field]:
            raise ValueError(f"{name}: MTEB dataset {field} does not match manifest")
    if task.is_multilingual:
        raise ValueError("This protocol supports the eight monolingual retrieval tasks only")
    load_pinned_test(task, entry["source"])
    selected = select_qrels(entry, {"test": task.relevant_docs["test"]}, "test")
    if not set(selected) <= set(task.queries["test"]):
        raise ValueError(f"{name}: query text missing for held-out query")
    if any(d not in task.corpus["test"] for docs in selected.values() for d in docs):
        raise ValueError(f"{name}: corpus document missing for held-out judgment")
    task.queries["test"] = {q: task.queries["test"][q] for q in selected}
    task.relevant_docs["test"] = selected
    return task


def evaluation_context(cfg: dict, tasks: list[str], manifest_path=None):
    """Keep model/cache identity, but isolate result directories by manifest checksum."""
    from csrv2_repro.common import REPO_ROOT, RESULTS

    cfg = {**cfg, "tasks": tasks}
    path = manifest_path or cfg.get("split_manifest")
    out = RESULTS / cfg["exp_id"] / cfg["run_name"]
    if not path:
        return cfg, None, out
    path = (REPO_ROOT / path).resolve()
    manifest = load_manifest(path)
    missing = set(tasks) - manifest["tasks"].keys()
    if missing:
        raise ValueError(f"Tasks absent from manifest: {sorted(missing)}")
    cfg["split_manifest"] = str(path)
    cfg["evaluation"] = {
        "protocol": PROTOCOL,
        "manifest_sha256": manifest["sha256"],
        "paper_comparable": False,
        "released_checkpoint_training_overlap": "unknown",
        "query_counts": {t: manifest["tasks"][t]["partitions"]["test"]["n_queries"] for t in tasks},
    }
    # Full-test paper targets must not appear as comparable targets in held-out summaries.
    cfg.pop("paper_target_mean", None)
    cfg.pop("paper_target_mean_by_k", None)
    return cfg, manifest, out / f"heldout-{manifest['sha256']}"
