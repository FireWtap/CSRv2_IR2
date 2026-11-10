import io

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from csrv2_repro.upstream_sae import (
    OriginalCSR,
    VerifiedCSR,
    annealed_k,
    linear_loss,
    make_optimizer,
    supervised_contrastive,
    upstream_modules,
)


def model(auxk=4):
    return VerifiedCSR(n_inputs=5, n_latents=16, topk=2, auxk=auxk, dead_threshold=3)


def batch():
    torch.manual_seed(41)
    return {
        "embeddings": torch.randn(4, 5, requires_grad=True),
        "keys": [("task", "query", str(i)) for i in range(4)],
        "positive_mask": torch.tensor(
            [
                [False, True, False, False],
                [True, False, False, False],
                [False, False, False, True],
                [False, False, True, False],
            ]
        ),
    }


def test_original_label_key_bug_and_vanishing_supervision():
    zoo, trainer = upstream_modules()
    dataset = zoo.CustomDataset({"data": np.ones((4, 5)), "label": np.arange(4)})
    labels = torch.tensor([dataset[i][1] for i in range(4)])
    assert labels.tolist() == [0, 0, 0, 0]
    z = torch.eye(4, requires_grad=True)
    loss = trainer.label_based_cl(z, z, labels)
    loss.backward()
    assert loss.item() == 0
    assert torch.count_nonzero(z.grad) == 0


def test_reuses_original_methods_and_preserves_projection_and_gradients():
    torch.manual_seed(6)
    original = OriginalCSR(16, 2, 4, False, 5, 3)
    repaired = model()
    repaired.load_state_dict(original.state_dict(), strict=True)
    assert VerifiedCSR.__init__ is not OriginalCSR.__init__
    for method in ("forward", "csr_forward", "decode", "encode_pre_act", "preprocess"):
        assert getattr(VerifiedCSR, method) is getattr(OriginalCSR, method)
    assert repaired.decoder.linear is repaired.encoder
    assert len(list(repaired.parameters())) == 3
    x = torch.randn(4, 5)
    original.eval()
    repaired.eval()
    old_outputs, new_outputs = original(x), repaired(x)
    for i in (0, 2, 3, 4):
        torch.testing.assert_close(old_outputs[i], new_outputs[i], rtol=0, atol=0)
    old_outputs[3].square().mean().backward()
    new_outputs[3].square().mean().backward()
    for p, q in zip(original.parameters(), repaired.parameters()):
        torch.testing.assert_close(p.grad, q.grad, rtol=0, atol=0)
    assert (repaired.encode(x).count_nonzero(dim=1) <= 2).all()


def test_topk_does_not_mutate_input_and_auxiliary_uses_only_dead_latents():
    repaired = model()
    repaired.eval()
    repaired.stats_last_nonzero[0] = 10
    x = torch.arange(1.0, 17).repeat(2, 1).requires_grad_()
    before = x.detach().clone()
    _, auxiliary = repaired.top_k(x)
    torch.testing.assert_close(x, before)
    assert auxiliary[:, 0].eq(1).all()
    assert auxiliary[:, 1:].eq(0).all()
    auxiliary.sum().backward()
    assert x.grad[:, 0].eq(1).all()
    assert x.grad[:, 1:].eq(0).all()


def test_original_mutates_preactivations_and_counts_each_forward():
    original = OriginalCSR(16, 2, 4, False, 5, 3)
    original.eval()
    x = torch.ones(2, 16)
    original.top_k(x)
    assert x.eq(0).all()
    assert original.stats_last_nonzero.eq(1).all()


def test_auxiliary_disabled_and_activity_counter_once_per_batch():
    repaired = model(auxk=0)
    data = batch()
    linear_loss(repaired, data, 2)
    assert repaired.stats_last_nonzero.max() == 1
    before = repaired.stats_last_nonzero.clone()
    repaired.encode(data["embeddings"])
    repaired.eval()
    repaired(data["embeddings"])
    assert torch.equal(before, repaired.stats_last_nonzero)


def test_schedule_is_original_and_reaches_target_at_seventy_percent():
    original = upstream_modules()[1].get_current_k
    for step in range(101):
        assert annealed_k(step, 64, 2, 100, 10240) == original(step, 64, 2, 100)
    assert annealed_k(0, 64, 2, 100, 10240) == 64
    assert annealed_k(70, 64, 2, 100, 10240) == 2
    with pytest.raises(ValueError, match="capacity"):
        annealed_k(0, 16384, 4096, 100, 10240)
    with pytest.raises(ValueError, match="capacity"):
        annealed_k(0, 10240, 2560, 100, 10240)


@pytest.mark.parametrize("multiple_positives", [False, True])
def test_supervision_matches_printed_equation_and_gradients(multiple_positives):
    z = torch.tensor([[1.0, 0], [0.6, 0.4], [0.0, 1], [0.1, 0.9]], requires_grad=True)
    mask = batch()["positive_mask"]
    if multiple_positives:
        mask[0, 2] = mask[2, 0] = True
    actual = supervised_contrastive(z, mask)
    scores = (z @ z.T).exp()
    expected = -torch.stack(
        [(scores[i, mask[i]].sum() / scores[i, torch.arange(4) != i].sum()).log() for i in range(4)]
    ).mean()
    torch.testing.assert_close(actual, expected)
    actual_gradient = torch.autograd.grad(actual, z, retain_graph=True)[0]
    torch.testing.assert_close(actual_gradient, torch.autograd.grad(expected, z)[0])
    assert actual_gradient.abs().sum() > 0
    if not multiple_positives:
        original = upstream_modules()[1].label_based_cl(z, z, torch.tensor([0, 0, 1, 1]))
        assert not torch.isclose(actual, original)


def test_missing_positive_anchors_have_zero_contribution_not_nan():
    data = batch()
    mask = data["positive_mask"].clone()
    mask[2:, :] = False
    z = torch.eye(4, requires_grad=True)
    loss = supervised_contrastive(z, mask)
    torch.testing.assert_close(loss, torch.tensor(0.5 * np.log(3), dtype=z.dtype))
    loss.backward()
    assert torch.isfinite(z.grad).all()
    with pytest.raises(ValueError, match="no distinct"):
        supervised_contrastive(z, torch.zeros_like(mask))
    with pytest.raises(ValueError, match="diagonal"):
        supervised_contrastive(z, torch.eye(4, dtype=torch.bool))


def test_loss_terms_step_and_checkpoint_parity():
    data = batch()
    repaired = model()
    repaired.stats_last_nonzero.fill_(10)
    losses = linear_loss(repaired, data, 2)
    z = repaired.encode(data["embeddings"])
    reconstruction = repaired.decode(z)
    _, auxiliary = repaired.top_k(repaired.encode_pre_act(data["embeddings"]), topk=8)
    expected_aux = F.mse_loss(
        repaired.decoder(auxiliary), (data["embeddings"] - reconstruction).detach()
    )
    torch.testing.assert_close(losses["auxiliary"], expected_aux)
    torch.testing.assert_close(
        losses["total"],
        losses["reconstruction"]
        + losses["reconstruction_4k"] / 8
        + losses["auxiliary"] * 0.1
        + losses["supervised"],
    )
    optimizer = make_optimizer(repaired, lr=2e-5, weight_decay=1e-4)
    before = repaired.encoder.weight.detach().clone()
    optimizer.zero_grad()
    losses["total"].backward()
    assert data["embeddings"].grad is None
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in repaired.parameters())
    optimizer.step()
    assert not torch.equal(before, repaired.encoder.weight)
    saved = io.BytesIO()
    torch.save(repaired.state_dict(), saved)
    saved.seek(0)
    restored = model()
    restored.load_state_dict(torch.load(saved, weights_only=True), strict=True)
    torch.testing.assert_close(
        restored.encode(data["embeddings"]), repaired.encode(data["embeddings"]), rtol=0, atol=0
    )
    assert restored.decoder.linear is restored.encoder


def test_rejects_duplicate_rows_and_capacity_overflow():
    data = batch()
    repaired = model()
    data["keys"][1] = data["keys"][0]
    with pytest.raises(ValueError, match="distinct"):
        linear_loss(repaired, data, 2)
    with pytest.raises(ValueError, match="capacity"):
        linear_loss(repaired, batch(), 8)
