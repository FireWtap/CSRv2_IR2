"""E3: audit existing embeddings and build ID-indexed training data without model inference."""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
from pathlib import Path

import numpy as np

from csrv2_repro.common import (
    OUTPUTS,
    REPO_ROOT,
    RESULTS,
    load_config,
    provenance,
    resolve_dtype,
    upstream_prompts,
    write_json,
)
from csrv2_repro.e3_missing_queries import file_sha256
from csrv2_repro.splits import digest, load_manifest, load_source, load_source_qrels, select_qrels

logger = logging.getLogger(__name__)


def cache_path(
    root: Path, task: str, side: str, texts: list[str], model_tag: str, prompt: str | None
) -> Path:
    """Recompute CachedEncoder's key from actual ordered texts, never infer IDs from row counts."""
    h = hashlib.sha1()
    for value in (model_tag, prompt or "", *texts):
        h.update(value.encode("utf-8"))
        h.update(b"\x00")
    return root / task / f"{side}-{len(texts)}-{h.hexdigest()[:16]}.npy"


def audit_shard(path: Path, ids: list[str], texts: list[str], dim: int) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"No cache matching pinned texts/model/prompt/order: {path}")
    values = np.load(path, mmap_mode="r", allow_pickle=False)
    if values.shape != (len(ids), dim) or values.dtype != np.float32:
        raise ValueError(f"Unexpected embedding shape/dtype in {path}")
    for start in range(0, len(ids), 2048):
        chunk = values[start : start + 2048]
        if not np.isfinite(chunk).all() or np.any(np.linalg.norm(chunk, axis=1) == 0):
            raise ValueError(f"Nonfinite/zero embeddings in {path}")
    return {
        "path": str(path.resolve()),
        "shape": list(values.shape),
        "dtype": str(values.dtype),
        "sha256": file_sha256(path),
        "ids_sha256": digest(ids),
        "texts_sha256": digest(texts),
    }


def text_map(rows, corpus=False) -> dict[str, str]:
    texts = {}
    for row in rows:
        key = str(row["_id"])
        if key in texts:
            raise ValueError(f"Duplicate source ID: {key}")
        texts[key] = row.get("title", "") + " " + row["text"] if corpus else row["text"]
    return texts


def build_task(
    name: str,
    entry: dict,
    source_qrels: dict,
    queries: dict,
    corpus: dict,
    cache_root: Path,
    model_tag: str,
    query_prompt: str | None,
    document_prompt: str | None,
    dim: int,
    chunk_size: int,
) -> dict:
    if chunk_size < 1:
        raise ValueError("corpus_chunk_size must be positive")
    # MTEB 1.38.39 says 'length' in its log but actually sorts string document IDs descending.
    doc_ids = sorted(corpus, reverse=True)
    shards = []
    for start in range(0, len(doc_ids), chunk_size):
        ids = doc_ids[start : start + chunk_size]
        texts = [corpus[d] for d in ids]
        path = cache_path(cache_root, name, "document", texts, model_tag, document_prompt)
        shard = audit_shard(path, ids, texts, dim)
        shards.append({**shard, "start": start, "stop": start + len(ids)})
    if not shards:
        raise ValueError(f"Empty corpus: {name}")

    partitions = {}
    full_test_shard = None
    for partition in ("train", "validation"):
        qrels = select_qrels(entry, source_qrels, partition)
        ids = list(qrels)
        if not set(ids) <= queries.keys():
            raise ValueError(f"Missing query texts: {name}/{partition}")
        missing = {
            q: {d: grade for d, grade in docs.items() if d not in corpus}
            for q, docs in qrels.items()
            if any(d not in corpus for d in docs)
        }
        without_positive = [
            q
            for q, docs in qrels.items()
            if not any(d in corpus and grade > 0 for d, grade in docs.items())
        ]
        if missing:
            logger.warning(
                "%s/%s: preserving %d judgments for absent corpus documents; "
                "%d queries have no available positive",
                name,
                partition,
                sum(len(d) for d in missing.values()),
                len(without_positive),
            )
        if any(not any(score > 0 for score in docs.values()) for docs in qrels.values()):
            raise ValueError(f"Query has no positive judgment: {name}/{partition}")
        shard, rows = None, []
        if ids:
            texts = [queries[q] for q in ids]
            path = cache_path(cache_root, name, "query", texts, model_tag, query_prompt)
            if path.exists():
                shard = audit_shard(path, ids, texts, dim)
                rows = list(range(len(ids)))
            elif entry["strategy"] == "fallback":
                if full_test_shard is None:
                    # MTEB preserves the source queries' order after filtering by test qrels.
                    test_ids = [q for q in queries if q in source_qrels["test"]]
                    if set(test_ids) != set(source_qrels["test"]):
                        raise ValueError(f"Missing source test query texts: {name}")
                    test_texts = [queries[q] for q in test_ids]
                    test_path = cache_path(
                        cache_root, name, "query", test_texts, model_tag, query_prompt
                    )
                    full_test_shard = audit_shard(test_path, test_ids, test_texts, dim)
                    test_rows = {q: i for i, q in enumerate(test_ids)}
                shard = full_test_shard
                rows = [test_rows[q] for q in ids]
            else:
                raise FileNotFoundError(f"Missing official {partition} query embeddings: {path}")
        partitions[partition] = {
            "query_ids": ids,
            "rows": rows,
            "shard": shard,
            "qrels": qrels,
            "missing_corpus_judgments": missing,
            "queries_without_available_positives": without_positive,
        }
    return {
        "schema_version": 1,
        "task": name,
        "source": entry["source"],
        "corpus": {"ids": doc_ids, "shards": shards},
        "partitions": partitions,
    }


def summarize_task(index: dict) -> dict:
    counts = {}
    corpus_ids = set(index["corpus"]["ids"])
    for part, spec in index["partitions"].items():
        membership = {}
        for docs in spec["qrels"].values():
            for did, grade in docs.items():
                if grade > 0 and did in corpus_ids:
                    membership[did] = membership.get(did, 0) + 1
        counts[part] = {
            "queries": len(spec["query_ids"]),
            "judgments": sum(len(v) for v in spec["qrels"].values()),
            "positive_pairs": sum(membership.values()),
            "positive_judgments": sum(
                v > 0 for docs in spec["qrels"].values() for v in docs.values()
            ),
            "missing_corpus_judgments": sum(
                len(d) for d in spec["missing_corpus_judgments"].values()
            ),
            "queries_without_available_positives": len(spec["queries_without_available_positives"]),
            "positive_documents": len(membership),
            "documents_shared_by_queries": sum(n > 1 for n in membership.values()),
            "grades": sorted({v for docs in spec["qrels"].values() for v in docs.values()}),
        }
    return {
        "corpus_documents": len(index["corpus"]["ids"]),
        "corpus_shards": len(index["corpus"]["shards"]),
        "partitions": counts,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/E3_training_data.toml")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    cfg = load_config(REPO_ROOT / args.config)
    dense = resolve_dtype(load_config(REPO_ROOT / cfg["dense_config"]), cfg["dtype"])
    cfg = {**cfg, "resolved_dense_config": dense}
    manifest = load_manifest(REPO_ROOT / cfg["split_manifest"])
    root = Path(os.environ.get("SCRATCH_ROOT", OUTPUTS))
    cache_root = root / "emb_cache" / dense["run_name"]
    model_tag = "|".join(
        str(v)
        for v in (
            dense["model"],
            dense.get("revision"),
            dense["dtype"],
            dense.get("max_seq_length", 512),
            dense.get("padding_side", "left"),
        )
    )
    identity = digest({"manifest_sha256": manifest["sha256"], "config": cfg})
    directory = root / "training_data" / cfg["run_name"] / identity
    prompts = upstream_prompts()
    summary = {
        "status": "auditing",
        "manifest_sha256": manifest["sha256"],
        "tasks": {},
        "provenance": provenance(cfg),
        "source_code_sha256": {
            f: file_sha256(Path(__file__).parent / f)
            for f in ("build_training_data.py", "splits.py", "training_data.py")
        },
    }
    output = RESULTS / "E3" / cfg["run_name"] / "summary.json"
    bundle = {
        "schema_version": 1,
        "manifest_sha256": manifest["sha256"],
        "config": cfg,
        "model_tag": model_tag,
        "embedding_dim": cfg["embedding_dim"],
        "pool": "partition queries plus full unlabeled corpus (transductive)",
        "tasks": {},
    }
    for name in cfg["tasks"]:
        logger.info("Auditing %s against pinned source texts", name)
        entry = manifest["tasks"][name]
        qrels = load_source_qrels(entry["source"])
        queries = text_map(next(iter(load_source(entry["source"], "queries").values())))
        corpus = text_map(next(iter(load_source(entry["source"], "corpus").values())), corpus=True)
        query_prompt = prompts[name] if dense["prompt_mode"] != "none" else None
        document_prompt = prompts[name] if dense["prompt_mode"] == "both" else None
        index = build_task(
            name,
            entry,
            qrels,
            queries,
            corpus,
            cache_root,
            model_tag,
            query_prompt,
            document_prompt,
            cfg["embedding_dim"],
            cfg["corpus_chunk_size"],
        )
        index["manifest_sha256"] = manifest["sha256"]
        path = directory / f"{name}.json"
        write_json(path, index)
        bundle["tasks"][name] = {"index": path.name, "sha256": file_sha256(path)}
        summary["tasks"][name] = summarize_task(index)
        write_json(output, summary)
        logger.info("%s: verified %s", name, summary["tasks"][name])
    write_json(directory / "split_manifest.json", manifest)
    bundle["sha256"] = digest(bundle)
    write_json(directory / "dataset.json", bundle)
    summary.update(
        status="complete",
        dataset_path=str((directory / "dataset.json").resolve()),
        dataset_sha256=bundle["sha256"],
    )
    write_json(output, summary)
    print(directory / "dataset.json")


if __name__ == "__main__":
    main()
