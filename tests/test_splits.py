from copy import deepcopy
from types import SimpleNamespace

import pytest

from csrv2_repro import splits
from csrv2_repro.common import run_mteb, write_json
from csrv2_repro.prepare_splits import export_qrels

SOURCE = {"path": "test/retrieval", "revision": "abc123"}


def source_qrels():
    # Shared documents, multiple positives and a zero judgment must survive partitioning.
    return {"test": {f"q{i:02}": {f"d{i}": 2, "shared": 1, "irrelevant": 0} for i in range(20)}}


def manifest_for(qrels=None):
    entry = splits.build_task_split(qrels or source_qrels(), SOURCE)
    return splits.seal_manifest({"Toy": entry})


def fake_task(qrels=None):
    qrels = qrels or source_qrels()
    return SimpleNamespace(
        metadata=SimpleNamespace(name="Toy", dataset=SOURCE),
        is_multilingual=False,
        data_loaded=True,
        queries={"test": {qid: qid for qid in qrels["test"]}},
        relevant_docs=deepcopy(qrels),
        corpus={
            "test": {
                **{f"d{i}": f"document {i}" for i in range(20)},
                "shared": "shared positive",
                "irrelevant": "distractor",
                "unjudged": "unjudged distractor",
            }
        },
    )


def test_fallback_is_query_disjoint_complete_and_order_independent():
    original = source_qrels()
    shuffled = {"test": dict(reversed(list(original["test"].items())))}
    a, b = manifest_for(original), manifest_for(shuffled)
    assert a == b
    entry = a["tasks"]["Toy"]
    train = splits.select_qrels(entry, original, "train")
    test = splits.select_qrels(entry, original, "test")
    assert len(train) == 16 and len(test) == 4
    assert train.keys().isdisjoint(test)
    assert {**train, **test} == original["test"]
    assert all(docs["shared"] == 1 and docs["irrelevant"] == 0 for docs in train.values())
    assert entry != splits.build_task_split(original, SOURCE, seed=43)


def test_official_train_dev_test_are_preserved():
    source = {"train": {"tr": {"a": 1}}, "dev": {"v": {"a": 2}}, "test": {"te": {"a": 1}}}
    entry = splits.build_task_split(source, SOURCE, validation_fraction=0.5)
    assert entry["strategy"] == "official"
    for output, original in [("train", "train"), ("validation", "dev"), ("test", "test")]:
        assert splits.select_qrels(entry, source, output) == source[original]


@pytest.mark.parametrize("official", [False, True])
def test_optional_validation_only_uses_training_pool(official):
    source = source_qrels()
    if official:
        source["train"] = {f"train{i}": {"shared": 1} for i in range(20)}
    original = splits.build_task_split(source, SOURCE)
    entry = splits.build_task_split(source, SOURCE, validation_fraction=0.25)
    assert entry["partitions"]["test"] == original["partitions"]["test"]
    train = set(entry["partitions"]["train"]["query_ids"])
    val = set(entry["partitions"]["validation"]["query_ids"])
    assert train.isdisjoint(val)
    assert train | val == set(original["partitions"]["train"]["query_ids"])


def test_manifest_and_source_drift_fail_closed():
    manifest = manifest_for()
    entry = manifest["tasks"]["Toy"]
    changed = source_qrels()
    changed["test"]["q00"]["shared"] = 3
    with pytest.raises(ValueError, match="Qrels changed"):
        splits.select_qrels(entry, changed, "train")
    manifest["seed"] = 1
    with pytest.raises(ValueError, match="checksum"):
        splits.validate_manifest(manifest)
    overlap = source_qrels()
    overlap["train"] = {"q00": {"different-doc": 1}}
    with pytest.raises(ValueError, match="overlap"):
        splits.build_task_split(overlap, SOURCE)


def test_many_to_many_rows_and_conflicting_labels():
    rows = [
        {"query-id": q, "corpus-id": d, "score": s}
        for q, docs in source_qrels()["test"].items()
        for d, s in docs.items()
    ]
    assert splits.qrels_from_rows(reversed(rows)) == source_qrels()["test"]
    with pytest.raises(ValueError, match="Conflicting"):
        splits.qrels_from_rows([rows[0], {**rows[0], "score": 3}])


def test_mteb_filter_keeps_full_corpus_and_only_heldout_queries():
    task = fake_task()
    corpus = task.corpus
    manifest = manifest_for()
    splits.configure_mteb_task(task, manifest)
    task.load_data(eval_splits=["test"])
    task.load_data()  # MTEB may call this again; it must be idempotent.
    expected = manifest["tasks"]["Toy"]["partitions"]["test"]["query_ids"]
    assert task.corpus is corpus and len(task.corpus["test"]) == 23
    assert sorted(task.queries["test"]) == expected
    assert sorted(task.relevant_docs["test"]) == expected
    assert "d0" in task.corpus["test"] and "unjudged" in task.corpus["test"]


def test_pinned_loader_preserves_mteb_text_construction(monkeypatch):
    task = fake_task()
    task.data_loaded = False
    calls = []

    def load(source, config="default"):
        calls.append((source, config))
        if config == "corpus":
            return {"corpus": [{"_id": 4, "title": "Title", "text": "Body"}]}
        if config == "queries":
            return {"queries": [{"_id": 7, "text": "Question"}, {"_id": 8, "text": "Unused"}]}
        return {"test": [{"query-id": 7, "corpus-id": 4, "score": 2}]}

    monkeypatch.setattr(splits, "load_source", load)
    splits.load_pinned_test(task, SOURCE)
    assert task.corpus == {"test": {"4": "Title Body"}}
    assert task.queries == {"test": {"7": "Question"}}
    assert task.relevant_docs == {"test": {"7": {"4": 2}}}
    assert calls == [(SOURCE, c) for c in ("default", "corpus", "queries")]


def test_hf_cache_fallback_cannot_silently_use_another_revision(monkeypatch, tmp_path):
    import datasets

    monkeypatch.setenv("SCRATCH_ROOT", str(tmp_path))
    cached = SimpleNamespace(cache_files=[{"filename": "/cache/abc123/data.arrow"}])
    calls = []

    def load(*args, **kwargs):
        calls.append(kwargs)
        return {"test": cached}

    monkeypatch.setattr(datasets, "load_dataset", load)
    splits.load_source(SOURCE)
    assert calls[0]["revision"] == "abc123"
    cached.cache_files = [{"filename": "/cache/different/data.arrow"}]
    with pytest.raises(ValueError, match="cache does not match"):
        splits.load_source(SOURCE)


def test_legacy_config_loads_jsonl_from_exact_snapshot(monkeypatch, tmp_path):
    import datasets
    import huggingface_hub

    path = tmp_path / "snapshots" / SOURCE["revision"] / "queries.jsonl"
    seen = []

    def download(repo, filename, **kwargs):
        assert repo == SOURCE["path"] and filename == "queries.jsonl"
        assert kwargs["revision"] == SOURCE["revision"]
        return str(path)

    def load(*args, **kwargs):
        if args[0] != "json":
            raise ValueError("BuilderConfig 'queries' not found. Available: ['default']")
        seen.append(kwargs["data_files"])
        return {"queries": [{"_id": "q", "text": "query"}]}

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    monkeypatch.setattr(datasets, "load_dataset", load)
    assert splits.load_source(SOURCE, "queries")["queries"][0]["_id"] == "q"
    assert seen == [{"queries": str(path)}]


def test_training_exports_and_api_share_manifest(tmp_path, monkeypatch):
    manifest = manifest_for()
    checksums = export_qrels(manifest, {"Toy": source_qrels()}, tmp_path)
    monkeypatch.setattr(splits, "load_source_qrels", lambda _: source_qrels())
    assert len(checksums) == 3
    assert splits.load_manifest(tmp_path / "manifest.json") == manifest
    assert len((tmp_path / "Toy/train.jsonl").read_text().splitlines()) == 16 * 3
    assert len(splits.training_qrels(manifest, "Toy")) == 16
    assert splits.training_qrels(manifest, "Toy", "validation") == {}
    with pytest.raises(ValueError, match="only train or validation"):
        splits.training_qrels(manifest, "Toy", "test")


def test_context_separates_results_but_preserves_embedding_cache_identity(tmp_path):
    path = tmp_path / "manifest.json"
    manifest = manifest_for()
    write_json(path, manifest)
    cfg = {"exp_id": "E1", "run_name": "dense", "paper_target_mean": 58.37}
    resolved, selected, out = splits.evaluation_context(cfg, ["Toy"], path)
    assert selected == manifest
    assert resolved["run_name"] == "dense"
    assert "paper_target_mean" not in resolved
    assert resolved["evaluation"]["paper_comparable"] is False
    assert out.name == "heldout-" + manifest["sha256"]
    assert splits.evaluation_context(cfg, ["Toy"])[1] is None
    with pytest.raises(ValueError, match="absent"):
        splits.evaluation_context(cfg, ["Unknown"], path)


def test_run_mteb_uses_filtered_tasks_and_refuses_result_mixing(tmp_path, monkeypatch):
    import mteb

    monkeypatch.setattr(mteb, "get_tasks", lambda **kwargs: [fake_task()])
    observed = []

    class Evaluation:
        def __init__(self, tasks):
            self.tasks = tasks

        def run(self, model, **kwargs):
            for task in self.tasks:
                task.load_data()
                observed.append((len(task.queries["test"]), len(task.corpus["test"])))
            write_json(tmp_path / "Toy.json", {"scores": {"test": [{"ndcg_at_10": 0.5}]}})

    monkeypatch.setattr(mteb, "MTEB", Evaluation)
    result = run_mteb(None, ["Toy"], tmp_path, False, {}, manifest_for())
    assert result["mean"] == 50 and observed == [(4, 23)]
    with pytest.raises(ValueError, match="protocol mismatch"):
        run_mteb(None, ["Toy"], tmp_path, True, {})
    with pytest.raises(ValueError, match="protocol mismatch"):
        other = splits.seal_manifest(
            {"Toy": splits.build_task_split(source_qrels(), SOURCE, seed=43)}, seed=43
        )
        run_mteb(None, ["Toy"], tmp_path, False, {}, other)


@pytest.mark.parametrize("problem", ["revision", "query", "document"])
def test_invalid_evaluation_inputs_are_rejected(problem):
    task, manifest = fake_task(), manifest_for()
    if problem == "revision":
        task.metadata = SimpleNamespace(name="Toy", dataset={**SOURCE, "revision": "different"})
    elif problem == "query":
        task.queries["test"] = {}
    else:
        task.corpus["test"] = {}
    with pytest.raises(ValueError):
        splits.filter_mteb_task(task, manifest)


def test_real_mteb_scores_only_heldout_queries_against_full_corpus(tmp_path, monkeypatch):
    """Offline end-to-end MTEB smoke test: real metric/ranking/prediction writer, toy vectors."""
    import json

    import mteb
    import numpy as np
    from sentence_transformers.util import cos_sim

    task = mteb.get_task("SciFact")
    qrels = {"test": {f"q{i:02}": {f"d{i:02}": 1} for i in range(20)}}
    entry = splits.build_task_split(qrels, task.metadata.dataset)
    manifest = splits.seal_manifest({"SciFact": entry})
    task.queries = {"test": {q: q for q in qrels["test"]}}
    task.corpus = {"test": {f"d{i:02}": f"d{i:02}" for i in range(21)}}
    task.relevant_docs = qrels
    task.data_loaded = True
    monkeypatch.setattr(mteb, "get_tasks", lambda **kwargs: [task])
    seen = []

    class Encoder:
        mteb_model_meta = mteb.get_model_meta("sentence-transformers/all-MiniLM-L6-v2").model_copy(
            update={"name": "test/heldout", "revision": "local"}
        )

        def encode(self, sentences, **kwargs):
            seen.extend(sentences)
            return np.eye(21, dtype=np.float32)[[int(s[1:]) for s in sentences]]

        def similarity(self, a, b):
            return cos_sim(a, b)

    result = run_mteb(Encoder(), ["SciFact"], tmp_path, False, {}, manifest)
    assert result == {"per_task": {"SciFact": 100.0}, "mean": 100.0}
    heldout = set(entry["partitions"]["test"]["query_ids"])
    assert set(s for s in seen if s.startswith("q")) == heldout
    assert set(s for s in seen if s.startswith("d")) == set(task.corpus["test"])
    prediction_path = next(tmp_path.rglob("SciFact_default_predictions.json"))
    assert set(json.loads(prediction_path.read_text())) == heldout
