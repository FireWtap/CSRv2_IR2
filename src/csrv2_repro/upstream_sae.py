"""Verified upstream SAE components with local repairs (see sae_audit.md).

This is the single-device CSRv2-linear training core, not a full-run recipe.
Only cached frozen backbone vectors enter it. No backbone or adapter is loaded.
"""

from __future__ import annotations

import importlib.util
import sys
from contextlib import contextmanager
from functools import lru_cache

import torch
from torch import Tensor
from torch.nn import functional as F

from .common import UPSTREAM_TEXT


@contextmanager
def _temporary_modules(modules):
    """Resolve upstream's bare imports without leaving generic module aliases behind."""
    previous = {name: sys.modules.get(name) for name in modules}
    sys.modules.update(modules)
    try:
        yield
    finally:
        for name, module in previous.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


def _load_file(name):
    spec = importlib.util.spec_from_file_location(
        f"_csrv2_original_{name}", UPSTREAM_TEXT / f"{name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@lru_cache(maxsize=1)
def upstream_modules():
    """Import actual pinned source; no copied model or annealing implementation."""
    model_zoo = _load_file("model_zoo")
    utils = _load_file("utils")
    with _temporary_modules({"model_zoo": model_zoo, "utils": utils}):
        trainer = _load_file("CSR_training")
    return model_zoo, trainer


OriginalCSR = upstream_modules()[0].CSR


def _topk_relu(x: Tensor, k: int) -> Tensor:
    if not 1 <= k <= x.shape[-1]:
        raise ValueError(f"k={k} outside latent capacity {x.shape[-1]}")
    values, indices = torch.topk(x, k, dim=-1)
    return torch.zeros_like(x).scatter(-1, indices, values).relu()


class VerifiedCSR(OriginalCSR):
    """Reuse upstream parameters, tied decoder, preprocessing and forward methods.

    Replace only top_k: do not mutate preactivations; allow auxk=0; count activity
    once per training batch (the target-k pass), never during evaluation/encode.
    State-dict keys and initialization are unchanged from upstream.
    """

    def __init__(self, *, n_inputs, n_latents, topk, auxk, dead_threshold=30):
        if not 1 <= topk <= n_latents or not 0 <= auxk <= n_latents:
            raise ValueError("topk/auxk outside latent capacity")
        if dead_threshold < 0:
            raise ValueError("dead_threshold must be nonnegative")
        super().__init__(n_latents, topk, auxk, False, n_inputs, dead_threshold)
        self.dead_threshold = dead_threshold

    def top_k(self, x: Tensor, topk=None):
        k = self.topk if topk is None else topk
        latents = _topk_relu(x, k)
        if self.training and k == self.topk:
            with torch.no_grad():
                active = (latents > 1e-5).any(dim=0)
                self.stats_last_nonzero.masked_fill_(active, 0).add_(1)
        auxiliary = torch.zeros_like(x)
        if self.auxk:
            dead = self.stats_last_nonzero > self.dead_threshold
            auxiliary = _topk_relu(x * dead.to(x.dtype), self.auxk)
        return latents, auxiliary

    def encode(self, x: Tensor, k: int | None = None) -> Tensor:
        """Inference transform with no normalization or activity-counter side effects."""
        return _topk_relu(self.encode_pre_act(x), self.topk if k is None else k)


def annealed_k(
    step: int,
    initial_k: int,
    final_k: int,
    total_steps: int,
    n_latents: int,
    decay_ratio: float = 0.7,
) -> int:
    """Use the original schedule, rejecting impossible 4k targets instead of clamping."""
    if step < 0 or total_steps < 1 or not 0 < decay_ratio <= 1:
        raise ValueError("invalid training step/decay settings")
    if not 1 <= final_k <= initial_k or 4 * initial_k > n_latents:
        raise ValueError("the complete k/4k schedule must fit the latent capacity")
    return upstream_modules()[1].get_current_k(step, initial_k, final_k, total_steps, decay_ratio)


def supervised_contrastive(z: Tensor, positives: Tensor) -> Tensor:
    """Printed Eq. 7: log of summed positive exp(dot), without extra self views.

    Uses D-009 group memberships. Nonpositive off-diagonal items are in-batch
    negatives, not asserted ground-truth negatives. Anchors lacking positives
    contribute zero; divide by the whole batch as printed in Eq. 7. A wholly
    unsupervised batch is rejected so supervision cannot silently disappear.
    """
    size = len(z)
    if positives.dtype != torch.bool or positives.shape != (size, size):
        raise ValueError("positive mask must be a boolean batch-by-batch matrix")
    if positives.device != z.device:
        raise ValueError("positive mask and embeddings must be on the same device")
    if positives.diagonal().any() or not torch.equal(positives, positives.T):
        raise ValueError("positive mask must be symmetric with an empty diagonal")
    anchors = positives.any(dim=1)
    if not anchors.any():
        raise ValueError("supervised batch has no distinct positive pairs")
    logits = z @ z.T
    diagonal = torch.eye(size, dtype=torch.bool, device=z.device)
    denominator = logits.masked_fill(diagonal, -torch.inf)[anchors].logsumexp(dim=1)
    numerator = logits.masked_fill(~positives, -torch.inf)[anchors].logsumexp(dim=1)
    return (denominator - numerator).sum() / size


def linear_loss(
    model: VerifiedCSR,
    batch: dict,
    k: int,
    *,
    aux_coefficient: float = 0.1,
    cl_coefficient: float = 1.0,
):
    """Eq. 8 with the printed dot-product SpSCL and unnormalized residual MSE.

    MSE uses upstream's elementwise mean convention. The auxiliary residual is
    a detached target, retaining upstream's stop-gradient on reconstruction.
    Duplicate keys must be handled by the eventual sampler, not made negatives.
    """
    if torch.distributed.is_initialized():
        raise RuntimeError("this audited core supports one process only")
    if 4 * k > model.encoder.out_features or k < 1:
        raise ValueError("k and 4k must both fit the latent capacity")
    if aux_coefficient < 0 or cl_coefficient < 0:
        raise ValueError("loss coefficients must be nonnegative")
    x = batch["embeddings"].detach()
    keys = batch["keys"]
    if len(keys) != len(x) or len(set(keys)) != len(keys):
        raise ValueError("batch must contain one key per distinct item")
    if x.ndim != 2 or x.shape[1] != model.encoder.in_features or len(x) == 0:
        raise ValueError("invalid dense embedding batch shape")
    if not torch.isfinite(x).all():
        raise ValueError("dense embeddings must be finite")
    model.topk = k
    _, _, z, reconstruction, auxiliary = model(x)
    _, _, _, reconstruction_4k, _ = model(x, topk=4 * k)
    losses = {
        "reconstruction": F.mse_loss(reconstruction, x),
        "reconstruction_4k": F.mse_loss(reconstruction_4k, x),
        "auxiliary": F.mse_loss(auxiliary - model.pre_bias, (x - reconstruction).detach())
        if model.auxk
        else z.sum() * 0,
        "supervised": supervised_contrastive(z, batch["positive_mask"])
        if cl_coefficient
        else z.sum() * 0,
    }
    losses["total"] = (
        losses["reconstruction"]
        + 0.125 * losses["reconstruction_4k"]
        + aux_coefficient * losses["auxiliary"]
        + cl_coefficient * losses["supervised"]
    )
    if not torch.isfinite(losses["total"]):
        raise FloatingPointError("nonfinite SAE objective")
    return losses


def make_optimizer(model: VerifiedCSR, *, lr: float, weight_decay: float):
    """Same Adam implementation and epsilon as the upstream trainer."""
    return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay, eps=6.25e-10)
