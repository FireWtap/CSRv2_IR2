"""Bounded E5 GPU pilots on frozen embeddings; no full training or test-set scoring."""

from __future__ import annotations

import argparse
import hashlib
import os
import time
from pathlib import Path

import numpy as np
import torch

from .common import REPO_ROOT, load_config, provenance, write_json
from .sae_sampling import PairFrequencySampler
from .training_data import RetrievalTrainingDataset, collate_training_items
from .upstream_sae import VerifiedCSR, annealed_k, linear_loss, make_optimizer


def _model(cfg, target, device):
    return VerifiedCSR(
        n_inputs=cfg["embedding_dim"],
        n_latents=cfg["hidden_dim"],
        topk=target,
        auxk=cfg["auxk"],
        dead_threshold=cfg["dead_threshold"],
    ).to(device)


def _loss(model, batch, k, cfg):
    has_positives = bool(batch["positive_mask"].any())
    return linear_loss(
        model,
        batch,
        k,
        aux_coefficient=cfg["beta"],
        cl_coefficient=cfg["gamma"] if has_positives else 0.0,
    )


def verify_checkpoint(path, model, optimizer, batch, cfg, target, device):
    """Verify saved inference at every requested k and one resumed optimizer step."""
    from sentence_transformers.sparse_encoder.models import SparseAutoEncoder

    saved = torch.load(path, map_location=device, weights_only=True)
    restored = _model(cfg, target, device)
    restored.load_state_dict(saved["state_dict"], strict=True)
    restored_optimizer = make_optimizer(restored, lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    restored_optimizer.load_state_dict(saved["optimizer"])
    evaluation = SparseAutoEncoder(
        input_dim=cfg["embedding_dim"],
        hidden_dim=cfg["hidden_dim"],
        k=target,
        k_aux=cfg["auxk"],
        normalize=False,
    ).to(device)
    evaluation.load_state_dict(saved["state_dict"], strict=True)
    model.eval()
    restored.eval()
    with torch.inference_mode():
        for k in cfg["evaluation_k"]:
            x = batch["embeddings"][:8]
            reference = model.encode(x, k)
            torch.testing.assert_close(restored.encode(x, k), reference, rtol=0, atol=0)
            st_output = evaluation({"sentence_embedding": x}, max_active_dims=k)[
                "sentence_embedding"
            ]
            torch.testing.assert_close(st_output, reference, rtol=0, atol=0)
    del evaluation, saved
    # This diagnostic continuation is not included in the saved pilot checkpoint.
    for candidate, opt in ((model, optimizer), (restored, restored_optimizer)):
        candidate.train()
        opt.zero_grad(set_to_none=True)
        _loss(candidate, batch, target, cfg)["total"].backward()
        opt.step()
    for key, value in model.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[key], value, rtol=0, atol=0)
    return {
        "inference_exact": True,
        "sentence_transformers_exact": True,
        "optimizer_resume_next_step_exact": True,
        "evaluation_k": cfg["evaluation_k"],
    }


def run_target(cfg, dataset, batches, target, output, source):
    torch.manual_seed(cfg["seed"])
    torch.cuda.manual_seed_all(cfg["seed"])
    device = torch.device("cuda:0")
    model = _model(cfg, target, device)
    optimizer = make_optimizer(model, lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    output.mkdir(parents=True, exist_ok=False)
    history, observed_tasks = [], set()
    baseline = model.encoder.weight.detach().clone()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for step, packed in enumerate(batches[: cfg["steps"]]):
        tick = time.perf_counter()
        batch = collate_training_items([dataset[i] for i in packed.indices])
        observed_tasks.update(key[0] for key in batch["keys"])
        batch["embeddings"] = batch["embeddings"].to(device)
        batch["positive_mask"] = batch["positive_mask"].to(device)
        excluded = torch.tensor(
            [not bool(g) for g in batch["positive_query_groups"]], device=device
        )
        k = annealed_k(step, cfg["initial_k"], target, cfg["steps"], cfg["hidden_dim"])
        observed = []
        hook = model.register_forward_hook(lambda module, args, result: observed.append(result[2]))
        try:
            losses = _loss(model, batch, k, cfg)
        finally:
            hook.remove()
        has_positives = bool(batch["positive_mask"].any())
        latent_norm = 0.0
        weight_gradient = None
        if has_positives:
            latent_gradient = torch.autograd.grad(
                losses["supervised"], observed[0], retain_graph=True
            )[0]
            if latent_gradient[excluded].count_nonzero():
                raise RuntimeError("supervised gradient on a reconstruction-only item")
            latent_norm = latent_gradient.norm().item()
            if step in {0, int(cfg["steps"] * 0.7), cfg["steps"] - 1}:
                weight_gradient = (
                    torch.autograd.grad(
                        losses["supervised"], model.encoder.weight, retain_graph=True
                    )[0]
                    .norm()
                    .item()
                )
                if not weight_gradient > 0:
                    raise RuntimeError("no supervised parameter gradient in pilot diagnostic batch")
        optimizer.zero_grad(set_to_none=True)
        losses["total"].backward()
        if not all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()):
            raise FloatingPointError("nonfinite/missing parameter gradient")
        optimizer.step()
        if not all(torch.isfinite(p).all() for p in model.parameters()):
            raise FloatingPointError("nonfinite parameter")
        torch.cuda.synchronize()
        history.append(
            {
                "step": step,
                "k": k,
                "rows": len(packed.indices),
                "pair_units": packed.positive_pairs,
                "directed_positives": batch["positive_mask"].sum().item(),
                "reconstruction_only_rows": excluded.sum().item(),
                "supervised_enabled": has_positives,
                "supervised_latent_gradient_norm": latent_norm,
                "supervised_weight_gradient_norm": weight_gradient,
                "losses": {key: value.item() for key, value in losses.items()},
                "wall_seconds": time.perf_counter() - tick,
            }
        )
        if step % 32 == 0 or step == cfg["steps"] - 1:
            print(
                f"k_target={target} step={step + 1}/{cfg['steps']} k={k} "
                f"loss={losses['total'].item():.6f}",
                flush=True,
            )
    training_seconds = time.perf_counter() - start
    training_peak = torch.cuda.max_memory_allocated()
    update_norm = (model.encoder.weight - baseline).norm().item()
    if not update_norm > 0 or observed_tasks != set(dataset.tasks):
        raise RuntimeError("pilot did not update weights or cover all tasks")
    del baseline
    checkpoint = output / "checkpoint.pth"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "global_step": cfg["steps"],
            "sampler_epoch": 0,
            "next_batch": cfg["steps"],
            "target_k": target,
            "config": cfg,
            "cpu_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state(),
            "dataset_sha256": dataset.dataset_sha256,
            "manifest_sha256": dataset.manifest_sha256,
        },
        checkpoint,
    )
    parity = verify_checkpoint(checkpoint, model, optimizer, batch, cfg, target, device)
    result = {
        "provenance": source,
        "target_k": target,
        "steps": len(history),
        "dataset_sha256": dataset.dataset_sha256,
        "manifest_sha256": dataset.manifest_sha256,
        "observed_tasks": sorted(observed_tasks),
        "checkpoint": str(checkpoint),
        "checkpoint_bytes": checkpoint.stat().st_size,
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "training_seconds": training_seconds,
        "training_peak_allocated_bytes": training_peak,
        "overall_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "encoder_update_norm": update_norm,
        "checkpoint_checks": parity,
        "no_supervised_gradient_on_reconstruction_only": True,
        "all_gradients_and_parameters_finite": True,
        "history": history,
        "retrieval_scores_measured": False,
        "paper_training_complete": False,
    }
    write_json(output / "summary.json", result)
    return {key: value for key, value in result.items() if key not in {"history", "provenance"}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    args.output.mkdir(parents=True)
    torch.set_num_threads(cfg["cpu_threads"])
    source = provenance(cfg)
    source["reviewed_source_commit"] = os.environ.get("PILOT_COMMIT")
    source["source_sha256"] = {
        str(p.relative_to(REPO_ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [
            Path(__file__),
            REPO_ROOT / "src/csrv2_repro/sae_sampling.py",
            REPO_ROOT / "src/csrv2_repro/upstream_sae.py",
            args.config.resolve(),
        ]
    }
    dataset = RetrievalTrainingDataset(Path(os.path.expandvars(cfg["dataset"])))
    if dataset.embedding_dim != cfg["embedding_dim"]:
        raise ValueError("embedding dimension mismatch")
    sampler = PairFrequencySampler(dataset, cfg["batch_size"], cfg["seed"])
    batches = list(sampler.batches(0))
    sampling = sampler.audit(batches)
    sampling.update(
        {
            "dataset_sha256": dataset.dataset_sha256,
            "manifest_sha256": dataset.manifest_sha256,
            "batch_order_sha256": hashlib.sha256(
                b"".join(np.asarray(b.indices, dtype="<i8").tobytes() + b"|" for b in batches)
            ).hexdigest(),
        }
    )
    write_json(args.output / "sampling.json", {"provenance": source, **sampling})
    print(f"Audited {sampling['row_appearances']} rows in {len(batches)} batches", flush=True)
    if args.audit_only:
        return
    if not torch.cuda.is_available():
        raise RuntimeError("GPU pilot requires CUDA; use --audit-only on login nodes")
    if cfg["steps"] > len(batches) or cfg["steps"] < 2:
        raise ValueError("pilot steps must fit the audited first epoch")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    source["numeric_policy"] = {
        "dtype": "float32",
        "tf32": False,
        "amp": False,
        "deterministic_algorithms": True,
    }
    results = []
    for target in cfg["target_k"]:
        results.append(
            run_target(cfg, dataset, batches, target, args.output / f"k{target}", source)
        )
        torch.cuda.empty_cache()
    write_json(
        args.output / "summary.json",
        {"provenance": source, "sampling": sampling, "pilots": results, "status": "passed"},
    )


if __name__ == "__main__":
    main()
