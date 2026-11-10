"""Reproduce upstream SAE defects and a small CPU integration check; no training run."""

import argparse
import hashlib
from pathlib import Path

import numpy as np
import torch

from csrv2_repro.common import REPO_ROOT, UPSTREAM_TEXT, provenance, write_json
from csrv2_repro.training_data import RetrievalTrainingDataset, collate_training_items
from csrv2_repro.upstream_sae import (
    VerifiedCSR,
    annealed_k,
    linear_loss,
    make_optimizer,
    supervised_contrastive,
    upstream_modules,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--review",
        action="store_true",
        help="64 CPU check steps across all tasks, including missing ArguAna positives",
    )
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(1227)
    zoo, trainer = upstream_modules()
    malformed = zoo.CustomDataset({"data": np.eye(4), "label": np.array([0, 0, 1, 1])})
    loaded_labels = torch.tensor([malformed[i][1] for i in range(4)])
    z = torch.eye(4, requires_grad=True)
    broken_loss = trainer.label_based_cl(z, z, loaded_labels)
    broken_gradient = torch.autograd.grad(broken_loss, z)[0]
    labels = torch.tensor([0, 0, 1, 1])
    positives = (labels[:, None] == labels[None, :]) & ~torch.eye(4, dtype=torch.bool)
    old_loss = trainer.label_based_cl(z, z, labels)
    printed_loss = supervised_contrastive(z, positives)
    assert broken_loss.item() == 0 and broken_gradient.count_nonzero() == 0
    assert not torch.isclose(old_loss, printed_loss)

    dataset = RetrievalTrainingDataset(args.dataset)
    items = []
    offset = 0
    for task in dataset.tasks:
        queries, docs = dataset.query_ids(task), dataset.document_ids(task)
        if args.review or task in {"ArguAna", "NFCorpus"}:
            available = set(docs)
            qid, did = next(
                (q, d)
                for q in queries
                for d, grade in dataset.judgments(task, q).items()
                if grade > 0 and d in available
            )
            items.extend(
                [
                    dataset[offset + queries.index(qid)],
                    dataset[offset + len(queries) + docs.index(did)],
                ]
            )
            if args.review and task == "ArguAna":
                missing_queries = [
                    q
                    for q in queries
                    if not any(
                        g > 0 and d in available for d, g in dataset.judgments(task, q).items()
                    )
                ]
                assert len(missing_queries) == 5
                items.extend(dataset[offset + queries.index(q)] for q in missing_queries)
        offset += len(queries) + len(docs)
    assert len(items) == (21 if args.review else 4)
    data = collate_training_items(items)
    excluded = torch.tensor([not bool(g) for g in data["positive_query_groups"]])
    steps = 64 if args.review else 1
    config = {
        "purpose": "CPU correctness check, not paper training or retrieval evaluation",
        "dataset": str(args.dataset.resolve()),
        "seed": 1227,
        "n_inputs": dataset.embedding_dim,
        "n_latents": 128,
        "topk": 2,
        "auxk": 16,
        "dead_threshold": 30,
        "steps": steps,
        "initial_topk": 16 if args.review else 2,
        "batch_size": len(items),
        "lr": 2e-5,
        "weight_decay": 1e-4,
        "aux_coefficient": 0.1,
        "cl_coefficient": 1.0,
        "loss_protocol": "printed_equations_D-011_D-012",
    }
    model = VerifiedCSR(n_inputs=dataset.embedding_dim, n_latents=128, topk=2, auxk=16)
    optimizer = make_optimizer(model, lr=2e-5, weight_decay=1e-4)
    before = model.encoder.weight.detach().clone()
    history = []
    for step in range(steps):
        k = annealed_k(step, config["initial_topk"], 2, steps, 128)
        observed = []
        hook = model.register_forward_hook(lambda module, args, output: observed.append(output[2]))
        try:
            losses = linear_loss(model, data, k)
        finally:
            hook.remove()
        supervised_grad, latent_grad = torch.autograd.grad(
            losses["supervised"], (model.encoder.weight, observed[0]), retain_graph=True
        )
        excluded_gradient = latent_grad[excluded].abs().sum().item()
        assert excluded_gradient == 0
        optimizer.zero_grad()
        losses["total"].backward()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters())
        assert supervised_grad.norm() > 0
        optimizer.step()
        assert all(torch.isfinite(p).all() for p in model.parameters())
        history.append(
            {
                "step": step,
                "k": k,
                "losses": {key: value.item() for key, value in losses.items()},
                "supervised_gradient_norm": supervised_grad.norm().item(),
                "reconstruction_only_supervised_gradient_l1": excluded_gradient,
                "max_inactivity_counter": model.stats_last_nonzero.max().item(),
            }
        )
    assert not torch.equal(before, model.encoder.weight)
    files = [UPSTREAM_TEXT / name for name in ("model_zoo.py", "CSR_training.py", "utils.py")] + [
        REPO_ROOT / "src/csrv2_repro/upstream_sae.py",
        REPO_ROOT / "tests/test_upstream_sae.py",
        Path(__file__).resolve(),
        REPO_ROOT / "docs/paper/latex_src/main.tex",
    ]
    report = {
        "provenance": provenance(config),
        "source_sha256": {
            str(p.relative_to(REPO_ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files
        },
        "dataset_sha256": dataset.dataset_sha256,
        "manifest_sha256": dataset.manifest_sha256,
        "original_loaded_labels": loaded_labels.tolist(),
        "original_broken_supervised_loss": broken_loss.item(),
        "original_broken_supervised_gradient_norm": broken_gradient.norm().item(),
        "correct_labels_original_loss": old_loss.item(),
        "printed_equation_loss_same_vectors": printed_loss.item(),
        "real_cached_batch_keys": data["keys"],
        "real_cached_batch_directed_positive_count": data["positive_mask"].sum().item(),
        "real_cached_batch_losses": {k: v.item() for k, v in losses.items()},
        "real_cached_batch_supervised_gradient_norm": supervised_grad.norm().item(),
        "real_cached_batch_encoder_update_norm": (before - model.encoder.weight).norm().item(),
        "reconstruction_only_count": excluded.sum().item(),
        "step_checks": history,
        "full_training_launched": False,
    }
    write_json(args.output, report)
    print(f"Audit and real-cache CPU integration passed; report: {args.output}")


if __name__ == "__main__":
    main()
