import json
from pathlib import Path

import numpy as np
import pytest

from csrv2_repro.build_training_data import audit_shard, build_task, cache_path
from csrv2_repro.common import write_json
from csrv2_repro.e3_missing_queries import file_sha256
from csrv2_repro.splits import build_task_split, digest, seal_manifest
from csrv2_repro.training_data import RetrievalTrainingDataset, collate_training_items


def vectors(texts):
    return np.array([[len(t), sum(map(ord, t)), 1] for t in texts], dtype=np.float32)


@pytest.fixture
def artifact(tmp_path):
    corpus = {"10": "very long body with title", "2": "short", "1": "other"}
    queries = {q: f"query {q}" for q in ("q4", "q2", "q0", "q3", "q1", "val", "test")}
    source = {"path": "toy/source", "revision": "pinned"}
    raw = {
        "Fallback": {"test": {f"q{i}": {"2": 2, "10": 1, "1": 0} for i in range(5)}},
        "Official": {
            "train": {"q0": {"2": 2, "10": 1, "1": 0}, "q1": {"2": 1}},
            "dev": {"val": {"1": 1}},
            "test": {"test": {"10": 1}},
        },
    }
    entries = {t: build_task_split(q, source) for t, q in raw.items()}
    manifest = seal_manifest(entries)
    cache = tmp_path / "cache"
    bundle = {
        "schema_version": 1,
        "manifest_sha256": manifest["sha256"],
        "embedding_dim": 3,
        "config": {"tasks": list(raw)},
        "tasks": {},
    }
    for task, qrels in raw.items():
        doc_ids = sorted(corpus, reverse=True)
        chunks = [[corpus[d] for d in doc_ids[i : i + 2]] for i in range(0, len(doc_ids), 2)]
        if task == "Fallback":
            qchunks = [[queries[q] for q in queries if q in qrels["test"]]]
        else:
            qchunks = [
                [queries[q] for q in entries[task]["partitions"][p]["query_ids"]]
                for p in ("train", "validation")
            ]
        for side, texts in [("document", x) for x in chunks] + [("query", x) for x in qchunks]:
            path = cache_path(cache, task, side, texts, "model", "prompt")
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, vectors(texts))
        index = build_task(
            task, entries[task], qrels, queries, corpus, cache, "model", "prompt", "prompt", 3, 2
        )
        index["manifest_sha256"] = manifest["sha256"]
        write_json(tmp_path / f"{task}.json", index)
        bundle["tasks"][task] = {
            "index": f"{task}.json",
            "sha256": file_sha256(tmp_path / f"{task}.json"),
        }
    write_json(tmp_path / "split_manifest.json", manifest)
    bundle["sha256"] = digest(bundle)
    write_json(tmp_path / "dataset.json", bundle)
    return tmp_path / "dataset.json", manifest, queries, corpus, raw


def test_corpus_and_query_vectors_follow_ids_across_shards(artifact):
    path, manifest, queries, corpus, _ = artifact
    data = RetrievalTrainingDataset(path)
    for task in data.tasks:
        assert data.document_ids(task) == ("2", "10", "1")  # lexical IDs, not lengths/numbers
        for did in data.document_ids(task):
            np.testing.assert_array_equal(
                data.document_vector(task, did), vectors([corpus[did]])[0]
            )
        for qid in data.query_ids(task):
            np.testing.assert_array_equal(data.query_vector(task, qid), vectors([queries[qid]])[0])
        assert data.query_ids(task) == tuple(
            manifest["tasks"][task]["partitions"]["train"]["query_ids"]
        )


def test_default_pool_excludes_validation_and_heldout_queries(artifact):
    path, manifest, _, corpus, _ = artifact
    data = RetrievalTrainingDataset(path)
    expected = {(t, "query", q) for t in data.tasks for q in data.query_ids(t)}
    expected |= {(t, "document", d) for t in data.tasks for d in corpus}
    items = [data[i] for i in range(len(data))]
    assert {x["key"] for x in items} == expected
    for task in data.tasks:
        for part in ("validation", "test"):
            for qid in manifest["tasks"][task]["partitions"][part]["query_ids"]:
                with pytest.raises(KeyError):
                    data.query_vector(task, qid)
    with pytest.raises(ValueError, match="only train or validation"):
        RetrievalTrainingDataset(path, "test")
    validation = RetrievalTrainingDataset(path, "validation")
    assert validation.query_ids("Fallback") == ()
    assert validation.query_ids("Official") == ("val",)
    assert validation.judgments("Official", "val") == {"1": 1}


def test_shared_positives_grades_and_task_namespaces_are_preserved(artifact):
    data = RetrievalTrainingDataset(artifact[0])
    assert data.judgments("Official", "q0") == {"2": 2, "10": 1, "1": 0}
    queries = [("Official", "q0"), ("Official", "q1")]
    docs = [("Official", "2"), ("Official", "10"), ("Official", "1"), ("Fallback", "2")]
    np.testing.assert_array_equal(
        data.relevance_matrix(queries, docs), [[2, 1, 0, -1], [1, -1, -1, -1]]
    )
    np.testing.assert_array_equal(
        data.positive_mask(queries, docs), [[True, True, False, False], [True, False, False, False]]
    )
    item = next(
        data[i] for i in range(len(data)) if data[i]["key"] == ("Official", "document", "2")
    )
    assert set(item["positive_query_groups"]) == {("Official", "q0"), ("Official", "q1")}
    assert ("Official", "q0", "1", 0) not in set(data.positive_pairs())
    # Validation labels are absent from this training view even for shared corpus documents.
    assert not any(
        group == ("Official", "val")
        for i in range(len(data))
        for group in data[i]["positive_query_groups"]
    )


def test_wrong_pinned_texts_or_prompt_refuse_cache_reuse(artifact):
    path, manifest, queries, corpus, raw = artifact
    args = [
        "Official",
        manifest["tasks"]["Official"],
        raw["Official"],
        queries,
        {**corpus, "2": "changed source text"},
        path.parent / "cache",
        "model",
        "prompt",
        "prompt",
        3,
        2,
    ]
    with pytest.raises(FileNotFoundError, match="matching pinned texts"):
        build_task(*args)
    args[4] = corpus
    args[8] = "wrong document prompt"
    with pytest.raises(FileNotFoundError, match="matching pinned texts"):
        build_task(*args)


def test_cache_key_matches_existing_encoder_contract(tmp_path):
    from types import SimpleNamespace
    from csrv2_repro.encoders import CachedEncoder
    from mteb.encoder_interface import PromptType

    owner = SimpleNamespace(cache_dir=tmp_path, model_tag="model")
    texts = ["first", "second"]
    assert cache_path(
        tmp_path, "Task", "document", texts, "model", "prompt"
    ) == CachedEncoder._cache_path(owner, texts, "prompt", "Task", PromptType.document)


def test_corrupted_array_is_rejected_and_returned_vectors_are_copies(artifact):
    path = artifact[0]
    data = RetrievalTrainingDataset(path)
    vector = data.document_vector("Official", "2")
    vector[:] = 0
    assert data.document_vector("Official", "2").any()
    index = json.loads((path.parent / "Official.json").read_text())
    shard_path = Path(index["corpus"]["shards"][0]["path"])
    values = np.load(shard_path)
    values[0, 0] += 1
    np.save(shard_path, values)
    with pytest.raises(ValueError, match="Embedding checksum"):
        RetrievalTrainingDataset(path)


@pytest.mark.parametrize("error", ["nan", "zero", "shape", "dtype"])
def test_shard_audit_rejects_invalid_vectors(tmp_path, error):
    values = np.ones((2, 3), dtype=np.float32)
    if error == "nan":
        values[0, 0] = np.nan
    if error == "zero":
        values[0] = 0
    if error == "shape":
        values = values[:, :2]
    if error == "dtype":
        values = values.astype(np.float64)
    path = tmp_path / "bad.npy"
    np.save(path, values)
    with pytest.raises(ValueError):
        audit_shard(path, ["a", "b"], ["a", "b"], 3)


def test_artifact_tampering_is_detected(artifact):
    path = artifact[0]
    index_path = path.parent / "Official.json"
    index = json.loads(index_path.read_text())
    index["partitions"]["train"]["query_ids"][0] = "test"
    write_json(index_path, index)
    with pytest.raises(ValueError, match="Task index checksum"):
        RetrievalTrainingDataset(path)


def test_collation_preserves_multiple_groups_without_transitive_merging(artifact):
    data = RetrievalTrainingDataset(artifact[0])
    lookup = {data[i]["key"]: data[i] for i in range(len(data))}
    keys = [
        ("Official", "query", "q0"),
        ("Official", "query", "q1"),
        ("Official", "document", "2"),
        ("Official", "document", "10"),
        ("Official", "document", "1"),
        ("Fallback", "document", "2"),
    ]
    batch = collate_training_items([lookup[k] for k in keys])
    assert tuple(batch["embeddings"].shape) == (6, 3)
    mask = batch["positive_mask"].numpy()
    assert mask[0, 2] and mask[1, 2] and mask[0, 3] and mask[2, 3]
    assert not mask[0, 1] and not mask[1, 3]  # Sharing a document does not merge queries.
    assert not mask.diagonal().any() and not mask[4].any() and not mask[5].any()
    repeated = collate_training_items([lookup[keys[0]], lookup[keys[0]]])
    assert not repeated["positive_mask"].any()


def test_absent_positive_documents_keep_queries_and_original_judgments(artifact):
    path, _, queries, corpus, raw = artifact
    qrels = raw["Official"]
    qrels["train"]["q0"] = {"absent-document": 2, "1": 0}
    source = {"path": "toy/source", "revision": "pinned"}
    entry = build_task_split(qrels, source)
    manifest = seal_manifest({"Official": entry})
    index = build_task(
        "Official",
        entry,
        qrels,
        queries,
        corpus,
        path.parent / "cache",
        "model",
        "prompt",
        "prompt",
        3,
        2,
    )
    index["manifest_sha256"] = manifest["sha256"]
    spec = index["partitions"]["train"]
    assert spec["missing_corpus_judgments"] == {"q0": {"absent-document": 2}}
    assert spec["queries_without_available_positives"] == ["q0"]
    write_json(path.parent / "Official.json", index)
    bundle = {
        "schema_version": 1,
        "manifest_sha256": manifest["sha256"],
        "embedding_dim": 3,
        "config": {"tasks": ["Official"]},
        "tasks": {
            "Official": {
                "index": "Official.json",
                "sha256": file_sha256(path.parent / "Official.json"),
            }
        },
    }
    bundle["sha256"] = digest(bundle)
    write_json(path, bundle)
    write_json(path.parent / "split_manifest.json", manifest)
    data = RetrievalTrainingDataset(path)
    assert data.query_ids("Official") == ("q0", "q1")
    assert data.judgments("Official", "q0") == {"absent-document": 2, "1": 0}
    assert data[0]["positive_query_groups"] == ()
    assert list(data.positive_pairs()) == [("Official", "q1", "2", 1)]
    assert not data.positive_mask([("Official", "q0")], [("Official", d) for d in corpus]).any()
    with pytest.raises(KeyError):
        data.document_vector("Official", "absent-document")
