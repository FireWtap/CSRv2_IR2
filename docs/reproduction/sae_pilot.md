# E5 sampling and GPU pilots

This implements the next step after the second SAE audit. It uses frozen, original
Qwen3-Embedding-4B vectors and the D-011/D-012 paper-equation core. It never loads a
backbone or LoRA adapter. Upstream remains read-only.

## Sampling (D-013)

The original preparer appends a query and document for each qrel. Uniformly iterating
our once-per-ID E3 representation would change those training frequencies substantially.
`PairFrequencySampler` instead builds virtual row appearances without copying vectors:

- Each available, positive **training** qrel contributes a query/document pair unit.
- Each item absent from those positive pairs appears once as a singleton. This includes
  all remaining corpus documents and the five missing-positive ArguAna queries.
- Query/document identities remain task-specific. Zero-grade judgments and held-out
  query judgments never create pair units or influence their frequencies.
- Pair units stay together. Distinct items sharing any query group are positives under
  D-009; no group is overwritten or transitively merged.

The available dataset gives 136,307 positive pair units and 214,074 singleton units:
**486,688 row appearances across 254,640 distinct items per epoch**. This differs from
the uncorrected upstream data where zero-grade/all-split judgments influenced selection.

The batch planner places pair units in descending maximum endpoint frequency, with
seeded tie ordering, into the least-filled compatible batch. It then fills slots with
singleton items and shuffles units within batches and full batches within the epoch.
This spreads frequent NFCorpus queries without putting two copies of one ID in a batch.
All appearances are retained; no example is duplicated beyond its prescribed frequency
or dropped to fill a batch. Partial batches are retained at the end and reported.

The initial randomized greedy packing was rejected during the audit because it left
hundreds of small residual batches. The frequency-first planner avoids that problem
on the actual E3 data. The exact batch sizes and order hash are saved in `sampling.json`.
This is an explicit change from upstream's independent row shuffle, motivated by useful
within-batch supervision. It is not claimed to recover the authors' original batch order.
Epoch ordering uses NumPy SeedSequence(seed=1227, epoch); the query split seed stays 42.

If constraints on another dataset require extra partial batches, they are reported,
not silently dropped. A batch with no positive pair uses reconstruction/auxiliary loss
only, with supervision disabled explicitly in its record. No synthetic self-positive
is introduced. Current pilot batches are checked for supervised gradients.

## Bounded pilot

`configs/E5_linear_pilot.toml` requests separate k=2 and k=16 heads, using identical
initial weights and the same first 256 batches. Both use:

| Parameter | Value |
|---|---|
| Input / latent dimensions | 2560 / 10240 |
| Batch size | 256 |
| Steps per target | 256 |
| Initial k | 64 |
| Annealing | Original linear schedule, reaches target at 70% of pilot steps |
| Adam learning rate / weight decay / epsilon | 2e-5 / 1e-4 / 6.25e-10 |
| Auxiliary k / coefficient | 1024 / 0.1 |
| Supervised coefficient | 1 |
| Precision | fp32, no AMP or TF32 |
| Hardware request | One A100, 20-minute SLURM limit |

This deliberately compressed schedule tests early/late sparse training in a short run.
It is not the first 256 steps of a 128-epoch paper run. Full-run epoch count and the
largest-k capacity conflict remain separate decisions; no full training is authorized
by successful pilot metrics alone. Batch size follows the paper rather than filling
all available GPU memory, since changing it changes the contrastive objective.

Each step checks finite losses, gradients and parameters, and zero supervised latent
gradients on reconstruction-only examples. Selected steps also verify nonzero supervised
encoder-weight gradients. All eight tasks must occur in the pilot. The saved checkpoint
includes weights, optimizer, configuration, next batch, global step, RNG states and E3
artifact identity. After saving, diagnostics compare inference exactly with the restored
model and Sentence Transformers at k=2,4,16,64,2560. A separate one-step continuation
checks that restored optimizer state gives the same update; that probe is not included
in the saved checkpoint's training-step count.

No held-out query is consumed and no test retrieval score is computed. The training
losses are diagnostics, not evidence of retrieval quality. Reported memory/time include
diagnostic overhead, including the initial-weight copy and selected extra gradient checks.

## Commands and artifacts

```bash
# Optional read-only CPU dataset/sampling audit (output directory must not exist).
PYTHONPATH="$PWD/src" uv run --frozen --no-sync python -m csrv2_repro.sae_pilot \
  --config configs/E5_linear_pilot.toml --audit-only --output outputs/E5_sampling_check

# Run from a clean, committed reproduction worktree.
mkdir -p outputs/slurm
PILOT_COMMIT=$(git rev-parse HEAD) sbatch --export=ALL jobs/snellius/E5_linear_pilot.job
```

The job checks that code/config/environment files still match the reviewed source
commit, allowing ledger/results-only commits while queued, and writes to
`$SCRATCH_ROOT/sae_pilots/E5-linear-<job-id>/`: full-epoch `sampling.json`, combined
`summary.json`, and one `k*/summary.json` plus `k*/checkpoint.pth` per target. Summary
files include source/config hashes, repository/upstream commits, SLURM id, dataset and
split-manifest hashes, per-step losses and resource use. Only small JSON evidence is
copied into `results/E5/`; checkpoints remain on scratch.

## Completed pilot: job 29072175

The job completed on 2026-11-10 at 21:11:28 Europe/Amsterdam, exit 0:0, after
8m07s on gcn42 (A100-SXM4-40GB). Both targets completed all 256 steps, using
batch size 256 throughout and covering all eight tasks. Finite-gradient checks,
reconstruction-only exclusion, exact inference parity with Sentence Transformers
at all five evaluation k values, and exact next-step optimizer restoration passed.
Checkpoint and source hashes, saved optimizer steps, finite weights/moments, tied
storage, full step histories, and sampler identities were independently checked
on 2026-11-11. Evidence: `results/E5/linear_pilot_fp32/verification.json`.

| Target k | Training-loop seconds | Median step seconds | Overall peak allocated GPU GiB |
|---|---:|---:|---:|
| 2 | 427.78 | 1.420 | 1.219 |
| 16 | 11.58 | 0.045 | 1.707 |

These are sequential runs over the same batches. Timing includes host data access
and first-use costs; cache and run-order effects were not isolated. The difference
does not establish that k=16 is intrinsically faster and is unsuitable for direct
full-run extrapolation. Peak memory includes checkpoint/evaluation diagnostics.
SLURM reported peak host RSS of 2,416,579 KiB (about 2.30 GiB).

No held-out retrieval scores were measured. Passing these pilots establishes that
the checked training paths execute correctly, not that training has converged or
that the paper's retrieval numbers have been reproduced. No full run was launched.

## Training target versus inference k

The two pilots are independent heads: one anneals 64 to 2, the other 64 to 16.
A single head can subsequently be evaluated at every supported inference k.
Training toward k=16 is not necessary to evaluate at k=16. It changes the terminal
objective: supervised features at 16 and reconstruction at 16/64, versus supervised
features at 2 and reconstruction at 2/8 for target 2. Earlier annealing exposure
does not make these terminal objectives identical. The multi-TopK reconstruction
loss promotes reuse across inference k, but does not guarantee equal performance
to a head optimized specifically for another target.

The k=16 pilot exercises a second sparsity regime; it does not settle the full-run
policy or require a separate full training for each reported inference k. A shared
target-2 head evaluated across all five k values is a valid experimental design;
matching the authors' table-generation policy still needs supporting evidence.
