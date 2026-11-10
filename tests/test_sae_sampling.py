from collections import Counter

import pytest

from csrv2_repro.sae_sampling import PairFrequencySampler


class ToyDataset:
    partition = "train"
    tasks = ("A", "B")

    def query_ids(self, task):
        return ("q0", "q1", "missing")

    def document_ids(self, task):
        return ("d0", "d1", "unlabelled")

    def positive_pairs(self):
        for task in self.tasks:
            yield task, "q0", "d0", 1
            yield task, "q0", "d1", 2
            yield task, "q1", "d0", 1

    def __len__(self):
        return 12


def test_pair_frequencies_singletons_and_distinct_batch_ids():
    sampler = PairFrequencySampler(ToyDataset(), batch_size=4)
    batches = list(sampler.batches())
    audit = sampler.audit(batches)
    assert audit["positive_pairs"] == 6
    assert audit["singleton_items"] == 4
    assert audit["row_appearances"] == 16
    counts = Counter(i for b in batches for i in b.indices)
    assert counts == Counter(
        {0: 2, 1: 1, 2: 1, 3: 2, 4: 1, 5: 1, 6: 2, 7: 1, 8: 1, 9: 2, 10: 1, 11: 1}
    )
    # Query/doc units are adjacent, stay intact, and are exhausted exactly once.
    pair_units = Counter(sampler.units[: sampler.pair_count])
    found = Counter()
    for b in batches:
        i = 0
        while i < len(b.indices):
            if i + 1 < len(b.indices) and tuple(b.indices[i : i + 2]) in pair_units:
                found[tuple(b.indices[i : i + 2])] += 1
                i += 2
            else:
                assert b.indices[i] in (2, 5, 8, 11)
                i += 1
    assert found == pair_units


def test_seed_epoch_determinism_and_no_dropped_tail():
    sampler = PairFrequencySampler(ToyDataset(), batch_size=5, seed=42)
    first = list(sampler.batches(0))
    assert first == list(sampler.batches(0))
    assert first != list(sampler.batches(1))
    for epoch in range(5):
        sampler.audit(list(sampler.batches(epoch)))


def test_high_frequency_query_does_not_loop_or_lose_pairs():
    class Star:
        partition = "train"
        tasks = ("A",)

        def query_ids(self, task):
            return ("q",)

        def document_ids(self, task):
            return tuple(str(i) for i in range(20))

        def positive_pairs(self):
            return (("A", "q", str(i), 1) for i in range(20))

        def __len__(self):
            return 21

    sampler = PairFrequencySampler(Star(), batch_size=8)
    batches = list(sampler.batches())
    assert len(batches) == 20
    assert sampler.audit(batches)["batch_size_counts"] == {2: 20}


def test_wrong_partition_and_missing_appearance_are_rejected():
    dataset = ToyDataset()
    dataset.partition = "validation"
    with pytest.raises(ValueError, match="training partition"):
        PairFrequencySampler(dataset)
    with pytest.raises(ValueError, match="batch size"):
        PairFrequencySampler(ToyDataset(), batch_size=1)
    sampler = PairFrequencySampler(ToyDataset(), batch_size=4)
    with pytest.raises(ValueError, match="frequencies"):
        sampler.audit(list(sampler.batches())[1:])


def test_pilot_checkpoint_resumes_optimizer_and_matches_evaluation(tmp_path):
    import torch
    from csrv2_repro.sae_pilot import _loss, _model, verify_checkpoint
    from csrv2_repro.upstream_sae import make_optimizer

    torch.manual_seed(42)
    cfg = {
        "embedding_dim": 5,
        "hidden_dim": 16,
        "auxk": 4,
        "dead_threshold": 3,
        "lr": 2e-5,
        "weight_decay": 1e-4,
        "beta": 0.1,
        "gamma": 1.0,
        "evaluation_k": [2, 4, 16],
    }
    labels = torch.tensor([0, 0, 1, 1])
    data = {
        "embeddings": torch.randn(4, 5),
        "keys": [("t", "query", str(i)) for i in range(4)],
        "positive_query_groups": [(("t", str(i // 2)),) for i in range(4)],
        "positive_mask": (labels[:, None] == labels[None, :]) & ~torch.eye(4, dtype=torch.bool),
    }
    model = _model(cfg, 2, "cpu")
    opt = make_optimizer(model, lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    for _ in range(2):
        opt.zero_grad()
        _loss(model, data, 2, cfg)["total"].backward()
        opt.step()
    checkpoint = tmp_path / "checkpoint.pth"
    torch.save({"state_dict": model.state_dict(), "optimizer": opt.state_dict()}, checkpoint)
    checks = verify_checkpoint(checkpoint, model, opt, data, cfg, 2, "cpu")
    assert checks["optimizer_resume_next_step_exact"]
    assert checks["sentence_transformers_exact"]
