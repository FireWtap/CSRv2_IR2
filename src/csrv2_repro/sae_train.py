"""E5: resumable single-head CSRv2-linear training on frozen cached embeddings."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import signal
import time
from pathlib import Path

import numpy as np
import torch

from .common import REPO_ROOT, load_config, provenance, write_json
from .sae_pilot import _loss, _model
from .sae_sampling import PairFrequencySampler
from .training_data import RetrievalTrainingDataset, collate_training_items
from .upstream_sae import annealed_k, make_optimizer


def order_hash(batches):
    digest = hashlib.sha256()
    for batch in batches:
        digest.update(np.asarray(batch.indices, dtype="<i8").tobytes())
        digest.update(b"|")
    return digest.hexdigest()


def prepare_schedule(sampler, epochs):
    """Audit every seeded epoch before fixing the global step budget.

    Packing may need different batch counts in different epochs. Keep only small
    metadata and regenerate each plan on demand, checking its identity before use.
    """
    if epochs < 1:
        raise ValueError("epochs must be positive")
    schedule = []
    for epoch in range(epochs):
        batches = list(sampler.batches(epoch))
        audit = sampler.audit(batches)
        audit["batch_size_counts"] = {str(k): v for k, v in audit["batch_size_counts"].items()}
        schedule.append({**audit, "order_sha256": order_hash(batches)})
        print(f"Audited epoch {epoch + 1}/{epochs}: {len(batches)} batches", flush=True)
    return schedule


def atomic_save(value, path):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        torch.save(value, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def empty_epoch_stats():
    return {"steps": 0, "rows": 0, "without_positives": 0, "loss_sums": {}}


def check_cursor(state, schedule):
    epoch, batch = state["sampler_epoch"], state["next_batch"]
    if not 0 <= epoch <= len(schedule):
        raise ValueError("invalid checkpoint epoch")
    if (epoch == len(schedule) and batch != 0) or (
        epoch < len(schedule) and not 0 <= batch < schedule[epoch]["batch_count"]
    ):
        raise ValueError("invalid checkpoint batch")
    expected = sum(s["batch_count"] for s in schedule[:epoch]) + batch
    if expected != state["global_step"] or len(state["completed_epochs"]) != epoch:
        raise ValueError("checkpoint cursor/global step mismatch")
    if state["epoch_stats"]["steps"] != batch:
        raise ValueError("checkpoint epoch statistics mismatch")


def train(cfg, dataset, sampler, schedule, output, identity, source, *,
          device="cuda:0", resume=False, should_stop=lambda: False, stop_after=None,
          epoch_timings=None):
    """Train one head; a bounded timing run keeps the full training schedule."""
    device = torch.device(device)
    total_steps = sum(s["batch_count"] for s in schedule)
    if len(schedule) != cfg["epochs"] or total_steps < 1:
        raise ValueError("schedule does not match configured epochs")
    annealed_k(0, cfg["initial_k"], cfg["target_k"], total_steps, cfg["hidden_dim"])
    torch.manual_seed(cfg["seed"])
    if device.type == "cuda":
        torch.cuda.manual_seed_all(cfg["seed"])
    model = _model(cfg, cfg["target_k"], device)
    optimizer = make_optimizer(model, lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    path = output / "checkpoint.pth"
    state = {
        "global_step": 0, "sampler_epoch": 0, "next_batch": 0,
        "epoch_stats": empty_epoch_stats(), "completed_epochs": [],
        "training_seconds": 0.0,
    }
    if resume:
        saved = torch.load(path, map_location="cpu", weights_only=True)
        if saved["identity"] != identity or saved["config"] != cfg:
            raise ValueError("checkpoint code/data/config identity mismatch")
        if saved["schedule"] != schedule:
            raise ValueError("checkpoint sampling schedule mismatch")
        state = saved["progress"]
        check_cursor(state, schedule)
        model.load_state_dict(saved["state_dict"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        torch.set_rng_state(saved["cpu_rng_state"])
        if device.type == "cuda":
            torch.cuda.set_rng_state(saved["cuda_rng_state"], device)
        del saved
    elif path.exists():
        raise FileExistsError("checkpoint exists; use --resume")
    model.train()
    start = time.monotonic()
    elapsed_before = state["training_seconds"]
    last_checkpoint = start

    def save():
        nonlocal last_checkpoint
        state["training_seconds"] = elapsed_before + time.monotonic() - start
        check_cursor(state, schedule)
        atomic_save({
            "state_dict": model.state_dict(), "optimizer": optimizer.state_dict(),
            "progress": state, "config": cfg, "identity": identity,
            "schedule": schedule, "provenance": source,
            "cpu_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
        }, path)
        last_checkpoint = time.monotonic()
        write_json(output / "progress.json", {
            "provenance": source, "identity": identity, "progress": state,
            "total_steps": total_steps,
            "status": "complete" if state["global_step"] == total_steps else "checkpointed",
            "next_k": annealed_k(state["global_step"], cfg["initial_k"], cfg["target_k"],
                                 total_steps, cfg["hidden_dim"]),
            "checkpoint": str(path), "retrieval_scores_measured": False,
        })
        print(f"Checkpoint: step {state['global_step']}/{total_steps}, "
              f"epoch {state['sampler_epoch']}, next batch {state['next_batch']}", flush=True)

    # Also make initialization recoverable if a signal arrives before the first step.
    if not resume:
        save()
    for epoch in range(state["sampler_epoch"], cfg["epochs"]):
        if should_stop() or (stop_after is not None and state["global_step"] >= stop_after):
            save()
            return state
        epoch_start = time.monotonic()
        batches = list(sampler.batches(epoch))
        if len(batches) != schedule[epoch]["batch_count"] or (
            order_hash(batches) != schedule[epoch]["order_sha256"]
        ):
            raise ValueError("regenerated batch order differs from audited schedule")
        for position in range(state["next_batch"], len(batches)):
            if should_stop() or (stop_after is not None and state["global_step"] >= stop_after):
                save()
                return state
            tick = time.monotonic()
            packed = batches[position]
            batch = collate_training_items([dataset[i] for i in packed.indices])
            for key in ("embeddings", "positive_mask"):
                batch[key] = batch[key].to(device)
            step = state["global_step"]
            k = annealed_k(step, cfg["initial_k"], cfg["target_k"], total_steps, cfg["hidden_dim"])
            optimizer.zero_grad(set_to_none=True)
            losses = _loss(model, batch, k, cfg)
            losses["total"].backward()
            if not all(p.grad is not None and torch.isfinite(p.grad).all()
                       for p in model.parameters()):
                raise FloatingPointError(f"nonfinite/missing gradient at step {step}")
            optimizer.step()
            if not all(torch.isfinite(p).all() for p in model.parameters()):
                raise FloatingPointError(f"nonfinite parameter at step {step}")
            values = {key: value.item() for key, value in losses.items()}
            stats = state["epoch_stats"]
            stats["steps"] += 1
            stats["rows"] += len(packed.indices)
            stats["without_positives"] += int(not batch["positive_mask"].any())
            for key, value in values.items():
                stats["loss_sums"][key] = stats["loss_sums"].get(key, 0.0) + value
            state["global_step"] += 1
            state["next_batch"] = position + 1
            if position + 1 == len(batches):
                state["completed_epochs"].append({
                    "epoch": epoch, **stats,
                    "mean_batch_losses": {key: value / stats["steps"]
                                          for key, value in stats["loss_sums"].items()},
                    "last_k": k,
                })
                state["sampler_epoch"] = epoch + 1
                state["next_batch"] = 0
                state["epoch_stats"] = empty_epoch_stats()
            if step % cfg["log_every_steps"] == 0 or position + 1 == len(batches):
                print(f"epoch={epoch + 1}/{cfg['epochs']} step={step + 1}/{total_steps} "
                      f"k={k} loss={values['total']:.6f} "
                      f"step_seconds={time.monotonic() - tick:.3f}", flush=True)
            if (state["global_step"] % cfg["checkpoint_every_steps"] == 0
                or time.monotonic() - last_checkpoint >= cfg["checkpoint_every_seconds"]
                or position + 1 == len(batches)):
                save()
        if epoch_timings is not None:
            epoch_timings.append({"epoch": epoch, "seconds": time.monotonic() - epoch_start})
    save()
    return state


def estimate_remaining(schedule, epoch_timings, startup_seconds):
    """Use a complete warm epoch, including loading, packing and checkpoint writes.

    Near-initial k is used as a conservative throughput proxy for the rest of the
    schedule. Dense matrix sizes stay fixed; the 25% margin covers later changes.
    Add the first-epoch excess again because continuation may land on a cold node.
    """
    if len(epoch_timings) != 2 or [x["epoch"] for x in epoch_timings] != [0, 1]:
        raise ValueError("calibration requires two complete initial epochs")
    seconds_per_step = epoch_timings[1]["seconds"] / schedule[1]["batch_count"]
    remaining_steps = sum(e["batch_count"] for e in schedule[2:])
    cold_restart_seconds = max(0.0, epoch_timings[0]["seconds"]
                               - schedule[0]["batch_count"] * seconds_per_step)
    estimate = startup_seconds + cold_restart_seconds + remaining_steps * seconds_per_step
    limit_minutes = math.ceil(estimate * 1.25 / 60)
    return {
        "method": "second-epoch throughput at near-initial k plus measured startup/cold excess",
        "epoch_timings": epoch_timings, "startup_seconds": startup_seconds,
        "cold_restart_seconds": cold_restart_seconds,
        "seconds_per_step": seconds_per_step, "remaining_steps": remaining_steps,
        "estimated_remaining_seconds": estimate, "safety_factor": 1.25,
        "slurm_limit_minutes": limit_minutes,
        "slurm_time": f"{limit_minutes // 60:02d}:{limit_minutes % 60:02d}:00",
    }


def main():
    startup_start = time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--calibrate", action="store_true",
                        help="train two real epochs, then estimate remaining walltime plus 25%%")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.calibrate and (args.resume or cfg["epochs"] < 3):
        raise ValueError("calibration requires a fresh run with at least three epochs")
    args.output.mkdir(parents=True, exist_ok=args.resume)
    # Prevent concurrent jobs from writing the same run, including during preparation.
    lock = (args.output / ".lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    stopping = []
    for signum in (signal.SIGUSR1, signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, lambda number, frame: stopping.append(number))
    torch.set_num_threads(cfg["cpu_threads"])
    if not torch.cuda.is_available():
        raise RuntimeError("full training requires a GPU")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    source = provenance(cfg)
    source["reviewed_source_commit"] = os.environ.get("TRAIN_COMMIT")
    source["numeric_policy"] = {"dtype": "float32", "amp": False, "tf32": False,
                                "deterministic_algorithms": True}
    paths = [REPO_ROOT / "src/csrv2_repro" / name for name in (
        "sae_train.py", "sae_pilot.py", "sae_sampling.py", "upstream_sae.py",
        "training_data.py", "splits.py", "common.py",
    )] + [args.config.resolve(), REPO_ROOT / "pyproject.toml", REPO_ROOT / "uv.lock"]
    source["source_sha256"] = {
        str(p.relative_to(REPO_ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths
    }
    write_json(args.output / f"attempt-{source['slurm_job_id']}-{time.time_ns()}.json", source)
    dataset = RetrievalTrainingDataset(Path(os.path.expandvars(cfg["dataset"])))
    if dataset.embedding_dim != cfg["embedding_dim"]:
        raise ValueError("embedding dimension mismatch")
    identity = {
        "dataset_sha256": dataset.dataset_sha256, "manifest_sha256": dataset.manifest_sha256,
        "source_sha256": source["source_sha256"], "upstream_commit": source["upstream_commit"],
    }
    sampler = PairFrequencySampler(dataset, cfg["batch_size"], cfg["seed"])
    # The batch script timestamp includes interpreter/import startup too.
    startup_seconds = (time.time() - float(os.environ["TRAIN_JOB_START_EPOCH"])
                       if "TRAIN_JOB_START_EPOCH" in os.environ
                       else time.monotonic() - startup_start)
    schedule_path = args.output / "schedule.json"
    if args.resume and schedule_path.exists():
        prepared = json.loads(schedule_path.read_text())
        if prepared["identity"] != identity or prepared["config"] != cfg:
            raise ValueError("schedule code/data/config identity mismatch")
        schedule = prepared["epochs"]
    else:
        schedule = prepare_schedule(sampler, cfg["epochs"])
        temporary = schedule_path.with_suffix(".tmp")
        write_json(temporary, {"provenance": source, "identity": identity,
                               "config": cfg, "epochs": schedule})
        os.replace(temporary, schedule_path)
    epoch_timings = []
    state = train(cfg, dataset, sampler, schedule, args.output, identity, source,
                  resume=args.resume and (args.output / "checkpoint.pth").exists(),
                  should_stop=lambda: bool(stopping), epoch_timings=epoch_timings,
                  stop_after=sum(e["batch_count"] for e in schedule[:2])
                  if args.calibrate else None)
    if args.calibrate and state["sampler_epoch"] == 2 and not stopping:
        estimate = estimate_remaining(schedule, epoch_timings, startup_seconds)
        write_json(args.output / "runtime_estimate.json", {
            "provenance": source, "identity": identity, **estimate,
        })
        print(f"Remaining estimate: {estimate['estimated_remaining_seconds'] / 3600:.2f} h; "
              f"+25% SLURM limit: {estimate['slurm_time']}", flush=True)
        return
    if state["sampler_epoch"] < cfg["epochs"]:
        # A saved time-limit interruption must not be mistaken for successful training.
        raise SystemExit(75)
    checkpoint = args.output / "checkpoint.pth"
    with checkpoint.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    write_json(args.output / "summary.json", {
        "provenance": source, "identity": identity, "progress": state,
        "status": "complete", "checkpoint": str(checkpoint), "checkpoint_sha256": digest,
        "peak_allocated_gpu_bytes": torch.cuda.max_memory_allocated(),
        "evaluation_k": cfg["evaluation_k"], "retrieval_scores_measured": False,
    })


if __name__ == "__main__":
    main()
