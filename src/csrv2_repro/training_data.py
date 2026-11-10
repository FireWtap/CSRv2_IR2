"""Read-only, ID-aware embedding dataset for E4/E5. No scalar-label conversion.

The reconstruction pool contains the selected partition's queries plus the full corpus.
Supervision comes exclusively from that partition's qrels. Query groups are namespaced
by task; a document may belong to multiple positive query groups without overwriting any.
"""

from __future__ import annotations

from bisect import bisect_right
import json
from pathlib import Path

import numpy as np

from csrv2_repro.e3_missing_queries import file_sha256
from csrv2_repro.splits import digest, load_manifest


def collate_training_items(items: list[dict]) -> dict:
    """Keep variable-length group memberships and build explicit within-batch positives.

    Two distinct items are positive when they share a positive training-query group.
    This does not merge query groups transitively through shared documents. Sampling
    is separate: a batch without a positive pair legitimately gets an all-false mask.
    """
    import torch

    if not items:
        raise ValueError("Cannot collate an empty batch")
    keys = [item["key"] for item in items]
    groups = [frozenset(item["positive_query_groups"]) for item in items]
    mask = np.zeros((len(items), len(items)), dtype=bool)
    for i, left in enumerate(groups):
        for j in range(i + 1, len(items)):
            mask[i, j] = mask[j, i] = keys[i] != keys[j] and bool(left & groups[j])
    return {
        "embeddings": torch.from_numpy(np.stack([item["embedding"] for item in items])),
        "positive_mask": torch.from_numpy(mask),
        "keys": keys,
        "positive_query_groups": [item["positive_query_groups"] for item in items],
    }


class RetrievalTrainingDataset:
    def __init__(self, path: str | Path, partition: str = "train"):
        if partition not in {"train", "validation"}:
            raise ValueError("Training dataset accepts only train or validation")
        path = Path(path)
        bundle = json.loads(path.read_text())
        if bundle.get("schema_version") != 1 or bundle.get("sha256") != digest(
            {k: v for k, v in bundle.items() if k != "sha256"}
        ):
            raise ValueError("Dataset index checksum/schema mismatch")
        manifest = load_manifest(path.parent / "split_manifest.json")
        if manifest["sha256"] != bundle["manifest_sha256"]:
            raise ValueError("Dataset split manifest mismatch")
        if set(bundle["tasks"]) != set(bundle["config"]["tasks"]):
            raise ValueError("Dataset task coverage mismatch")
        self.partition = partition
        self.manifest_sha256 = manifest["sha256"]
        self.dataset_sha256 = bundle["sha256"]
        self.embedding_dim = bundle["embedding_dim"]
        self.tasks = tuple(bundle["tasks"])
        self._data, self._arrays, self._fingerprints, self._ends = {}, {}, {}, []
        total = 0
        for name, location in bundle["tasks"].items():
            index_path = path.parent / location["index"]
            if file_sha256(index_path) != location["sha256"]:
                raise ValueError(f"Task index checksum mismatch: {name}")
            index = json.loads(index_path.read_text())
            entry = manifest["tasks"][name]
            if (
                index["task"] != name
                or index["source"] != entry["source"]
                or index["manifest_sha256"] != manifest["sha256"]
            ):
                raise ValueError(f"Task source/manifest mismatch: {name}")
            spec = index["partitions"][partition]
            if spec["query_ids"] != entry["partitions"][partition]["query_ids"]:
                raise ValueError(f"Query selection does not match manifest: {name}/{partition}")
            ids = spec["query_ids"]
            docs = index["corpus"]["ids"]
            if len(set(docs)) != len(docs) or not docs:
                raise ValueError(f"Duplicate/empty corpus IDs: {name}")
            if set(spec["qrels"]) != set(ids) or len(spec["rows"]) != len(ids):
                raise ValueError(f"Query/qrel coverage mismatch: {name}")
            if (
                sum(len(d) for d in spec["qrels"].values())
                != entry["partitions"][partition]["n_judgments"]
            ):
                raise ValueError(f"Judgment count mismatch: {name}")
            memberships = {d: [] for d in docs}
            missing = {
                q: {d: g for d, g in ds.items() if d not in memberships}
                for q, ds in spec["qrels"].items()
                if any(d not in memberships for d in ds)
            }
            without_positive = [
                q
                for q, ds in spec["qrels"].items()
                if not any(d in memberships and g > 0 for d, g in ds.items())
            ]
            if (
                missing != spec["missing_corpus_judgments"]
                or without_positive != spec["queries_without_available_positives"]
            ):
                raise ValueError(f"Missing-document audit mismatch: {name}")
            for qid, judged in spec["qrels"].items():
                for did, grade in judged.items():
                    if not isinstance(grade, int) or grade < 0:
                        raise ValueError(f"Invalid judgment: {name}/{qid}/{did}")
                    if grade > 0 and did in memberships:
                        memberships[did].append((name, qid))
            cursor = 0
            for shard in index["corpus"]["shards"]:
                self._open_shard(shard)
                if shard["start"] != cursor or shard["stop"] - cursor != shard["shape"][0]:
                    raise ValueError(f"Corpus shard gap/overlap: {name}")
                if digest(docs[cursor : shard["stop"]]) != shard["ids_sha256"]:
                    raise ValueError(f"Corpus shard ID ordering mismatch: {name}")
                cursor = shard["stop"]
            if cursor != len(docs):
                raise ValueError(f"Incomplete corpus shards: {name}")
            if ids:
                if spec["shard"] is None:
                    raise ValueError(f"Missing query shard: {name}")
                self._open_shard(spec["shard"])
                if len(set(spec["rows"])) != len(ids) or any(
                    not isinstance(row, int) or not 0 <= row < spec["shard"]["shape"][0]
                    for row in spec["rows"]
                ):
                    raise ValueError(f"Invalid query row selection: {name}")
            elif spec["shard"] is not None:
                raise ValueError(f"Unexpected shard for empty query partition: {name}")
            self._data[name] = {
                "queries": spec,
                "corpus": index["corpus"],
                "groups": memberships,
                "query_rows": dict(zip(ids, spec["rows"])),
                "document_rows": {d: i for i, d in enumerate(docs)},
                "shard_ends": [s["stop"] for s in index["corpus"]["shards"]],
            }
            total += len(ids) + len(docs)
            self._ends.append(total)

    def _open_shard(self, spec: dict):
        path = spec["path"]
        identity = (spec["sha256"], tuple(spec["shape"]), spec["dtype"])
        if path in self._fingerprints and self._fingerprints[path] != identity:
            raise ValueError(f"Conflicting embedding descriptors: {path}")
        if path not in self._arrays:
            if file_sha256(Path(path)) != spec["sha256"]:
                raise ValueError(f"Embedding checksum mismatch: {path}")
            values = np.load(path, mmap_mode="r", allow_pickle=False)
            if (
                list(values.shape) != spec["shape"]
                or values.dtype != np.float32
                or values.ndim != 2
                or values.shape[1] != self.embedding_dim
            ):
                raise ValueError(f"Invalid embedding shape/dtype: {path}")
            self._arrays[path] = values
            self._fingerprints[path] = identity

    def query_ids(self, task: str) -> tuple[str, ...]:
        return tuple(self._data[task]["queries"]["query_ids"])

    def document_ids(self, task: str) -> tuple[str, ...]:
        return tuple(self._data[task]["corpus"]["ids"])

    def query_vector(self, task: str, query_id: str) -> np.ndarray:
        spec = self._data[task]
        row = spec["query_rows"][query_id]  # Reject validation/test IDs absent from this view.
        return self._arrays[spec["queries"]["shard"]["path"]][row].copy()

    def document_vector(self, task: str, document_id: str) -> np.ndarray:
        spec = self._data[task]
        row = spec["document_rows"][document_id]
        shard = spec["corpus"]["shards"][bisect_right(spec["shard_ends"], row)]
        return self._arrays[shard["path"]][row - shard["start"]].copy()

    def judgments(self, task: str, query_id: str) -> dict[str, int]:
        return dict(self._data[task]["queries"]["qrels"][query_id])

    def positive_pairs(self):
        for task in self.tasks:
            for qid, docs in self._data[task]["queries"]["qrels"].items():
                for did, grade in docs.items():
                    if grade > 0 and did in self._data[task]["document_rows"]:
                        yield task, qid, did, grade

    def relevance_matrix(
        self, queries: list[tuple[str, str]], documents: list[tuple[str, str]]
    ) -> np.ndarray:
        """Original grades; -1 means unjudged, 0 means explicitly judged nonrelevant."""
        for task, did in documents:
            if did not in self._data[task]["document_rows"]:
                raise KeyError((task, did))
        matrix = np.full((len(queries), len(documents)), -1, dtype=np.int32)
        for i, (task, qid) in enumerate(queries):
            qrels = self.judgments(task, qid)
            for j, (doc_task, did) in enumerate(documents):
                if task == doc_task:
                    matrix[i, j] = qrels.get(did, -1)
        return matrix

    def positive_mask(
        self, queries: list[tuple[str, str]], documents: list[tuple[str, str]]
    ) -> np.ndarray:
        return self.relevance_matrix(queries, documents) > 0

    def __len__(self):
        return self._ends[-1] if self._ends else 0

    def __getitem__(self, position: int) -> dict:
        if position < 0:
            position += len(self)
        if not 0 <= position < len(self):
            raise IndexError(position)
        task_idx = bisect_right(self._ends, position)
        task = self.tasks[task_idx]
        offset = position - (self._ends[task_idx - 1] if task_idx else 0)
        spec = self._data[task]
        nq = len(spec["queries"]["query_ids"])
        if offset < nq:
            key = spec["queries"]["query_ids"][offset]
            vector = self.query_vector(task, key)
            groups = (
                ()
                if key in spec["queries"]["queries_without_available_positives"]
                else ((task, key),)
            )
            kind = "query"
        else:
            key = spec["corpus"]["ids"][offset - nq]
            vector = self.document_vector(task, key)
            groups, kind = tuple(spec["groups"][key]), "document"
        return {"key": (task, kind, key), "embedding": vector, "positive_query_groups": groups}
