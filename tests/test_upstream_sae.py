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
        "positive_query_groups": [(("task", str(i // 2)),) for i in range(4)],
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


def test_reconstruction_only_item_gets_no_supervised_gradient():
    data = batch()
    data["embeddings"] = torch.cat([data["embeddings"], torch.ones(1, 5)], dim=0)
    data["keys"].append(("task", "query", "missing-document"))
    data["positive_query_groups"] = [
        (("task", "a"),),
        (("task", "a"),),
        (("task", "b"),),
        (("task", "b"),),
        (),
    ]
    data["positive_mask"] = F.pad(data["positive_mask"], (0, 1, 0, 1))
    repaired = model()
    outputs = []
    hook = repaired.register_forward_hook(lambda module, args, output: outputs.append(output))
    try:
        losses = linear_loss(repaired, data, 2)
    finally:
        hook.remove()
    gradient = torch.autograd.grad(losses["supervised"], outputs[0][2])[0]
    assert gradient[-1].count_nonzero() == 0
    assert gradient[:-1].abs().sum() > 0


def test_supervised_finite_difference_and_excluded_candidate_invariance():
    torch.manual_seed(17)
    z = torch.randn(5, 3, dtype=torch.float64, requires_grad=True)
    positives = F.pad(batch()["positive_mask"], (0, 1, 0, 1))
    eligible = torch.tensor([True, True, True, True, False])

    def objective(values):
        return supervised_contrastive(values, positives, eligible=eligible)

    assert torch.autograd.gradcheck(objective, (z,), eps=1e-6, atol=1e-6, rtol=1e-4)
    changed = z.detach().clone()
    changed[-1] = 1000
    torch.testing.assert_close(objective(z), objective(changed), rtol=0, atol=0)
    torch.testing.assert_close(
        objective(z), supervised_contrastive(z[:4], positives[:4, :4]) * 4 / 5
    )


def test_complete_objective_and_parameter_gradients_against_independent_reference():
    data = batch()
    data["embeddings"] = data["embeddings"].detach().double()
    repaired = model().double()
    with torch.no_grad():
        repaired.pre_bias.copy_(torch.linspace(-0.2, 0.3, 5))
        repaired.latent_bias.copy_(torch.linspace(-0.1, 0.2, 16))
        repaired.stats_last_nonzero.fill_(10)
    weight, pre_bias, bias = [
        p.detach().clone().requires_grad_()
        for p in (repaired.encoder.weight, repaired.pre_bias, repaired.latent_bias)
    ]

    def select(values, k):
        # Independent ordering/masking implementation, not the production scatter helper.
        order = torch.argsort(values, dim=-1, descending=True)
        mask = torch.zeros_like(values, dtype=torch.bool)
        for i in range(len(values)):
            mask[i, order[i, :k]] = True
        return values.clamp_min(0) * mask

    x = data["embeddings"]
    pre = (x - pre_bias) @ weight.T + bias
    z, z4 = select(pre, 2), select(pre, 8)
    reconstruction, reconstruction4 = z @ weight + pre_bias, z4 @ weight + pre_bias
    dead = ~(z > 1e-5).any(dim=0)
    auxiliary = select(torch.where(dead, pre, 0.0), 4) @ weight
    assert auxiliary.count_nonzero() > 0
    scores = torch.exp(z @ z.T)
    supervised = -torch.stack(
        [
            torch.log(
                scores[i, data["positive_mask"][i]].sum()
                / sum(scores[i, j] for j in range(4) if j != i)
            )
            for i in range(4)
        ]
    ).mean()
    expected = {
        "reconstruction": ((reconstruction - x) ** 2).mean(),
        "reconstruction_4k": ((reconstruction4 - x) ** 2).mean(),
        "auxiliary": ((auxiliary - (x - reconstruction).detach()) ** 2).mean(),
        "supervised": supervised,
    }
    actual = linear_loss(repaired, data, 2)
    parameters = (repaired.encoder.weight, repaired.pre_bias, repaired.latent_bias)
    for term in expected:
        torch.testing.assert_close(actual[term], expected[term], rtol=1e-12, atol=1e-12)
        gradients = torch.autograd.grad(actual[term], parameters, retain_graph=True)
        reference = torch.autograd.grad(expected[term], (weight, pre_bias, bias), retain_graph=True)
        for g, r in zip(gradients, reference):
            torch.testing.assert_close(g, r, rtol=1e-10, atol=1e-12)


def test_dead_features_become_auxiliary_after_threshold_and_recover():
    repaired = VerifiedCSR(n_inputs=2, n_latents=4, topk=1, auxk=2, dead_threshold=3)
    pre = torch.tensor([[4.0, 3.0, 2.0, 1.0]])
    for step in range(1, 5):
        _, aux = repaired.top_k(pre)
        repaired.top_k(pre, topk=4)
        assert repaired.stats_last_nonzero.tolist() == [1, step, step, step]
        if step <= 3:
            assert aux.count_nonzero() == 0
        else:
            torch.testing.assert_close(aux, torch.tensor([[0.0, 3.0, 2.0, 0.0]]))
    repaired.top_k(torch.tensor([[1.0, 5.0, 2.0, 0.0]]))
    assert repaired.stats_last_nonzero.tolist() == [2, 1, 5, 5]


def test_inference_matches_sentence_transformers_evaluation_path():
    from sentence_transformers.sparse_encoder.models import SparseAutoEncoder

    repaired = model()
    evaluation = SparseAutoEncoder(input_dim=5, hidden_dim=16, k=2, k_aux=4)
    evaluation.load_state_dict(repaired.state_dict(), strict=True)
    x = batch()["embeddings"].detach()
    with torch.inference_mode():
        for k in (1, 2, 4, 16):
            actual = evaluation({"sentence_embedding": x.clone()}, max_active_dims=k)[
                "sentence_embedding"
            ]
            torch.testing.assert_close(actual, repaired.encode(x, k), rtol=0, atol=0)


def test_supervised_run_cannot_omit_group_memberships():
    data = batch()
    del data["positive_query_groups"]
    with pytest.raises(ValueError, match="query groups"):
        linear_loss(model(), data, 2)
