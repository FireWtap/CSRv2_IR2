# Shared query partitions (D-007)

`results/E3/split_manifest.json` is the common data contract for LoRA, SAE and held-out
evaluation. It records source repositories/revisions, qrel hashes and sorted query IDs.
`split_manifest_audit.json` records counts, export hashes and generation provenance.

| Task | Training queries | Validation queries | Evaluation queries |
|---|---:|---:|---:|
| ArguAna | 1,124 | 0 | 282 |
| NFCorpus | 2,590 | 324 | 323 |
| CQADupstackGamingRetrieval | 1,276 | 0 | 319 |
| CQADupstackUnixRetrieval | 857 | 0 | 215 |
| ClimateFEVERHardNegatives | 800 | 0 | 200 |
| FiQA2018 | 5,500 | 500 | 648 |
| SCIDOCS | 800 | 0 | 200 |
| SciFact | 809 | 0 | 300 |

NFCorpus, FiQA2018 and SciFact keep their official splits. The other five tasks use the
authors' 80/20 fallback ratio, with one shared **query-level** split. Sort unique query IDs,
shuffle indices with NumPy RandomState(42), assign the first ceil(0.2 * N) to evaluation.
All remaining queries belong to training. This does not claim to recover the authors' exact
historical assignments: their LoRA path splits qrel rows and their SAE path splits queries.
Grouping by query prevents different judgments for one query crossing the boundary.

Existing dev splits are preserved. The default creates no additional validation split.
To use early stopping or select hyperparameters for other tasks, first set a nonzero
`validation_fraction` in a separate split config and save a new manifest. This fraction
is taken from the training pool; test IDs remain identical. Never tune on the held-out test.

## Generate / verify the data contract

From the repository root (on Snellius, source `jobs/snellius/env.sh` first):

```bash
uv run --frozen python -m csrv2_repro.prepare_splits --config configs/E3_splits.toml
```

This downloads only pinned qrels, validates disjointness, and exports per-task
`train.jsonl`, `validation.jsonl`, `test.jsonl` plus a manifest/checksums under
`$SCRATCH_ROOT/data_splits/<manifest SHA>/` (or `outputs/data_splits/` locally).
Each line preserves `query-id`, `corpus-id`, and the original integer `score`, including
zero and graded relevance. Shared documents and multiple positives are retained.
Reruns reproduce the same manifest; replacing it with a different assignment requires
an explicit different `--output` path. `--export-dir` can choose another artifact location.

Future LoRA and SAE preparation must use these exports or the common API:

```python
from csrv2_repro.splits import load_manifest, training_qrels

manifest = load_manifest("results/E3/split_manifest.json")
qrels = training_qrels(manifest, "ArguAna", partition="train")
# qrels maps query ID -> {document ID: relevance grade}.
# Use only these query IDs when loading training query texts.
# Positive-pair sampling uses score > 0; zero grades are not positive pairs.
```

Apply caps/sampling **after** selecting the manifest partition. Never pass the original
full test qrels into the authors' preparation scripts, or add the held-out query texts back
as unsupervised singleton rows. Missing official training/validation query embeddings can
now be extracted with the E3 helper below; SAE training remains E4/E5 work.

Query separation is by ID within each task. It does not enforce document disjointness,
cross-task semantic deduplication, or absence of pretraining exposure. The full corpus is
the retrieval collection for every split; using its unlabeled documents in training must
be described as transductive. Validation/test relevance labels stay outside training.

## Evaluate the matching queries

All existing evaluation entry points accept `--split-manifest` (or `split_manifest` in TOML):

```bash
uv run --frozen python -m csrv2_repro.e1_dense_backbone --config configs/E1_dense_both.toml --dtype float32 --split-manifest results/E3/split_manifest.json
uv run --frozen python -m csrv2_repro.e2_mrl --config configs/E2_mrl_both.toml --dtype float32 --split-manifest results/E3/split_manifest.json
uv run --frozen python -m csrv2_repro.e0_released_ckpt --config configs/E6b_released_ckpt.toml --dtype float32 --split-manifest results/E3/split_manifest.json
```

`jobs/snellius/E3_heldout_eval.job` runs these three paths with a required `DTYPE` setting;
it has not been submitted. Select the agreed precision per D-005 before launching.

The loader checks dataset identities and qrel hashes, filters queries and qrels before
MTEB evaluation, and keeps **every corpus document**. MTEB 1.38.39's ordinary retrieval
loader ignores its metadata revision, so our held-out loader passes the pinned revision
explicitly while retaining MTEB's `title + " " + text` construction and metric code.

Scores and predictions go under `results/<experiment>/<run>/heldout-<manifest SHA>/`.
Cached Arrow data must also match the pinned revision: an offline/network-error fallback to
a different cached revision is rejected for qrels, queries and corpus alike.

Summaries include the protocol, checksum and per-task evaluated-query counts; paper targets
are cleared. Output markers reject reuse across protocols even with `--overwrite`.
The embedding cache retains its original identity: matching full-corpus embeddings can
be reused, while held-out query batches get distinct content-based cache keys.

Without the flag/config field, existing entry points still run the original full-test
reference protocol required by D-006. These results must not be mixed with held-out
retraining comparisons. New trained-head evaluators should pass the loaded manifest to
`common.run_mteb(..., split_manifest=manifest)` and use `evaluation_context` for output paths.
The released checkpoint remains an artifact reference: selecting our held-out IDs does
not establish that those queries were absent from its unknown training data.

## Verification

### Completing the dense query cache (D-008)

The shared E1 fp32 cache already contains every test query and the full corpus for all eight
tasks. For the five fallback tasks those query embeddings cover both manifest partitions.
Only NFCorpus (2,590 train + 324 validation), FiQA2018 (5,500 + 500), and SciFact (809 + 0)
need additional query encoding: 9,723 queries total. The E3 helper loads pinned query texts,
verifies the manifest qrels, and writes ordered IDs beside references to the new cached vectors.
It does not regenerate corpus embeddings or released-checkpoint embeddings.

```bash
source jobs/snellius/env.sh
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
uv run --frozen --no-sync python -m csrv2_repro.e3_missing_queries --prepare-only
# Use the printed JSON path for PREPARED_QUERIES:
PREPARED_QUERIES=/path/to/prepared-inputs.json sbatch --export=ALL jobs/snellius/E3_missing_queries.job
```

Preparation downloads data on the login node; the GPU job uses those frozen inputs offline.
The isolated checkout reuses the parent frozen venv with `--no-sync` and an explicit source path.
In a fresh checkout first run `uv sync --frozen`. Existing valid embedding files are reused.
`results/E3/missing_queries_fp32/summary.json` records the generating commit, config, SLURM job,
file checksums and paths to the ordered-ID indices under `$SCRATCH_ROOT/training_queries/`.
The vectors go into the existing `qwen3_4b_dense_prompt_both_fp32` cache. Both prompt variants
use the same query instructions, so these vectors can also serve the query-only variant.

The pinned SciFact snapshot has no HF configuration metadata for `queries`/`corpus`.
Our loader reads its JSONL files directly from the same verified snapshot in that case.

### Tests

```bash
uv run --frozen pytest -q tests
```

Tests cover query overlap, determinism, official splits, optional validation, graded and
many-to-many qrels, data/manifest drift, pinned loading, full-corpus retention, shared
training exports, and refusal to mix full-test/held-out results.
