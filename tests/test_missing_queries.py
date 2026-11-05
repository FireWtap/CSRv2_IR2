from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from csrv2_repro import e3_missing_queries as extraction
from csrv2_repro.splits import build_task_split, digest, seal_manifest


def inputs(monkeypatch):
    qrels = {
        "train": {"tr2": {"d": 1}, "tr1": {"d": 2}},
        "dev": {"val": {"d": 1}},
        "test": {"heldout": {"d": 1}},
    }
    entry = build_task_split(qrels, {"path": "test/dataset", "revision": "abc"})
    manifest = seal_manifest({"NFCorpus": entry})
    cfg = {"tasks": ["NFCorpus"], "partitions": ["train", "validation"]}
    monkeypatch.setattr(extraction, "load_source_qrels", lambda _: qrels)
    monkeypatch.setattr(
        extraction,
        "load_source",
        lambda *args: {
            "queries": [{"_id": q, "text": q + " text"} for q in ["heldout", "val", "tr2", "tr1"]]
        },
    )
    prepared = extraction.prepare_inputs(cfg, manifest, {})
    return cfg, manifest, prepared


def test_preparation_keeps_manifest_order_and_excludes_test(monkeypatch):
    cfg, manifest, prepared = inputs(monkeypatch)
    extraction.validate_prepared(prepared, cfg, manifest, {})
    assert prepared["parts"][0]["query_ids"] == ["tr1", "tr2"]
    assert prepared["parts"][0]["texts"] == ["tr1 text", "tr2 text"]
    assert prepared["parts"][1]["query_ids"] == ["val"]
    assert all("heldout" not in part["query_ids"] for part in prepared["parts"])
    with pytest.raises(ValueError, match="only accepts"):
        extraction.prepare_inputs({**cfg, "partitions": ["test"]}, manifest, {})


def test_prepared_tampering_and_incorrect_coverage_fail(monkeypatch):
    cfg, manifest, prepared = inputs(monkeypatch)
    altered = deepcopy(prepared)
    altered["parts"][0]["texts"][0] = "changed"
    with pytest.raises(ValueError, match="checksum"):
        extraction.validate_prepared(altered, cfg, manifest, {})
    altered = deepcopy(prepared)
    altered["parts"][0]["query_ids"] = ["heldout", "tr2"]
    altered["sha256"] = digest({k: v for k, v in altered.items() if k != "sha256"})
    with pytest.raises(ValueError, match="query order"):
        extraction.validate_prepared(altered, cfg, manifest, {})


def test_encoding_writes_index_and_reuses_cache(tmp_path, monkeypatch):
    from csrv2_repro.encoders import CachedEncoder

    cfg, manifest, prepared = inputs(monkeypatch)
    calls = []

    class Model:
        def get_sentence_embedding_dimension(self):
            return 3

        def encode(self, texts, **kwargs):
            calls.append((texts, kwargs))
            return np.ones((len(texts), 3), dtype=np.float32)

    # Exercise the real embedding cache without downloading a model for a unit test.
    encoder = object.__new__(CachedEncoder)
    encoder.model, encoder.model_tag = Model(), "test-model"
    encoder.prompts, encoder.prompt_mode = prepared["prompts"], "both"
    encoder.cache_dir, encoder.batch_size = tmp_path / "cache", 16
    part = prepared["parts"][0]
    record = extraction.encode_partition(
        encoder, part, tmp_path, manifest["sha256"], prepared["sha256"]
    )
    second = extraction.encode_partition(
        encoder, part, tmp_path, manifest["sha256"], prepared["sha256"]
    )
    assert len(calls) == 1 and calls[0][0] == ["tr1 text", "tr2 text"]
    assert calls[0][1]["prompt"] == prepared["prompts"]["NFCorpus"]
    assert record["cache_hit"] is False and second["cache_hit"] is True
    assert record["embedding_sha256"] == second["embedding_sha256"]
    assert record["shape"] == [2, 3]
    import json

    index = json.loads(Path(record["index_path"]).read_text())
    assert index["query_ids"] == part["query_ids"]
    assert index["manifest_sha256"] == manifest["sha256"]


def test_empty_partition_does_not_invoke_model(tmp_path):
    part = {"task": "SciFact", "partition": "validation", "query_ids": [], "texts": []}
    assert (
        extraction.encode_partition(None, part, tmp_path, "manifest", "prepared")["status"]
        == "empty"
    )
