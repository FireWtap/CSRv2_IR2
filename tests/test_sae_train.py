import json

import numpy as np
import pytest
import torch

from csrv2_repro.sae_sampling import PackedBatch
from csrv2_repro.sae_train import estimate_remaining, prepare_schedule, train
from csrv2_repro.upstream_sae import annealed_k


class Dataset:
    def __getitem__(self, index):
        return {
            "key": ("t", "query", str(index)),
            "embedding": np.random.default_rng(index).normal(size=5).astype("float32"),
            "positive_query_groups": (("t", str(index // 2)),) if index < 6 else (),
        }


class Sampler:
    # Exercise both variable epoch lengths and a reconstruction-only residual batch.
    def batches(self, epoch):
        if epoch % 2:
            return [PackedBatch((2, 3, 0, 1), 2), PackedBatch((4, 5), 1), PackedBatch((6,), 0)]
        return [PackedBatch((0, 1, 4, 5), 2), PackedBatch((2, 3, 6), 1)]

    def audit(self, batches):
        return {"batch_count": len(batches), "batch_size_counts": {len(b.indices): 1 for b in batches}}


@pytest.fixture
def cfg():
    return dict(seed=1227, epochs=4, embedding_dim=5, hidden_dim=16, auxk=4,
                initial_k=4, target_k=2, dead_threshold=3, lr=2e-5,
                weight_decay=1e-4, beta=.1, gamma=1., log_every_steps=100,
                checkpoint_every_steps=3, checkpoint_every_seconds=600)


def assert_nested_equal(left, right):
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_nested_equal(left[key], right[key])
    elif isinstance(left, list):
        assert len(left) == len(right)
        for x, y in zip(left, right):
            assert_nested_equal(x, y)
    else:
        assert left == right


@pytest.mark.parametrize("stop", [0, 1, 2, 4, 7])
def test_resume_exact_across_epochs_annealing_and_partial_batches(tmp_path, cfg, stop):
    torch.set_num_threads(1)
    sampler = Sampler()
    schedule = prepare_schedule(sampler, cfg["epochs"])
    assert sum(s["batch_count"] for s in schedule) == 10
    assert annealed_k(0, 4, 2, 10, 16) == 4
    assert annealed_k(7, 4, 2, 10, 16) == 2
    identity = {"dataset_sha256": "fixture", "source_sha256": "fixture"}
    full, interrupted = tmp_path / "full", tmp_path / "interrupted"
    full.mkdir()
    interrupted.mkdir()
    args = (cfg, Dataset(), sampler, schedule)
    train(*args, full, identity, {}, device="cpu")
    progress = train(*args, interrupted, identity, {}, device="cpu", stop_after=stop)
    assert progress["global_step"] == stop
    # Real CLI resumes from JSON schedule, not the original Python dictionary.
    restored_schedule = json.loads(json.dumps(schedule))
    train(cfg, Dataset(), sampler, restored_schedule, interrupted, identity, {},
          device="cpu", resume=True)
    a = torch.load(full / "checkpoint.pth", weights_only=True)
    b = torch.load(interrupted / "checkpoint.pth", weights_only=True)
    for key in ("state_dict", "optimizer", "cpu_rng_state", "schedule"):
        assert_nested_equal(a[key], b[key])
    a["progress"].pop("training_seconds")
    b["progress"].pop("training_seconds")
    assert a["progress"] == b["progress"]
    assert a["progress"]["global_step"] == 10
    assert a["progress"]["sampler_epoch"] == 4
    assert a["progress"]["next_batch"] == 0
    assert sum(e["rows"] for e in a["progress"]["completed_epochs"]) == 28
    assert sum(e["without_positives"] for e in a["progress"]["completed_epochs"]) == 2
    assert json.loads((interrupted / "progress.json").read_text())["status"] == "complete"


def test_resume_rejects_changed_identity_cursor_and_plan(tmp_path, cfg):
    sampler = Sampler()
    schedule = prepare_schedule(sampler, cfg["epochs"])
    args = (cfg, Dataset(), sampler, schedule, tmp_path)
    train(*args, {"source": "a"}, {}, device="cpu", should_stop=lambda: True)
    with pytest.raises(ValueError, match="identity mismatch"):
        train(*args, {"source": "b"}, {}, device="cpu", resume=True)
    path = tmp_path / "checkpoint.pth"
    state = torch.load(path, weights_only=True)
    state["progress"]["global_step"] = 3
    torch.save(state, path)
    with pytest.raises(ValueError, match="cursor/global step mismatch"):
        train(*args, {"source": "a"}, {}, device="cpu", resume=True)
    state["progress"]["global_step"] = 0
    state["schedule"][0]["order_sha256"] = "corrupted"
    torch.save(state, path)
    with pytest.raises(ValueError, match="sampling schedule mismatch"):
        train(*args, {"source": "a"}, {}, device="cpu", resume=True)


def test_runtime_estimate_uses_remaining_steps_and_exact_25_percent_margin():
    schedule = [{"batch_count": n} for n in [100, 100, 110, 90]]
    timing = [{"epoch": 0, "seconds": 400}, {"epoch": 1, "seconds": 100}]
    result = estimate_remaining(schedule, timing, startup_seconds=40)
    assert result["cold_restart_seconds"] == 300
    assert result["estimated_remaining_seconds"] == 540
    assert result["slurm_limit_minutes"] == 12 # 540 * 1.25 = 675 s, rounded up to a minute.
    assert result["slurm_time"] == "00:12:00"
    with pytest.raises(ValueError, match="two complete initial epochs"):
        estimate_remaining(schedule, timing[:1], 40)
