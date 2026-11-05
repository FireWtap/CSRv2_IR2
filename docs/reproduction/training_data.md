# E3 indexed embeddings and supervision

The builder audits existing dense fp32 vectors against the pinned source texts, then produces a
training dataset without copying/re-encoding embeddings. Configuration: `configs/E3_training_data.toml`.
Run from the isolated `CSRv2_IR2-E3` checkout, whose `.venv` reuses the frozen parent environment:

```bash
source jobs/snellius/env.sh
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
uv run --frozen --no-sync python -m csrv2_repro.build_training_data
```

In a fresh checkout, install the pinned environment with `uv sync --frozen` first. The command prints
the `dataset.json` path under `$SCRATCH_ROOT/training_data/training_data_fp32/`. Per-task indices and a
copy of the split manifest live alongside it. `results/E3/training_data_fp32/summary.json` records the
generating commit, config, source code hashes, corpus/query counts, grades and positive-pair counts.

## Cache audit

- Load qrels, query text and corpus text at the manifest's exact dataset revisions.
- Reproduce MTEB text construction (`title + " " + text`) and **descending string document-ID order**.
  MTEB's log incorrectly calls this a length sort. Preserve its 50,000-document chunks (FiQA has two).
- Recompute each cache key from model revision, inference dtype, length, padding, prompt and ordered
  text. A file with only a matching shape/count is insufficient. Mismatches fail; no inference runs.
- Verify shape, float32 dtype, finite/nonzero vectors and full-file SHA-256 hashes.
- Native train/validation queries use the E3 extraction files. Fallback tasks select explicit row
  indices from the existing full-test query arrays. The training view exposes only manifest train IDs;
  it never loops over every row of those source arrays or adds held-out queries as singleton examples.

This establishes a content match for the reused vectors even if the friend's cached HF repository
revision differs from the pinned revision. It verifies the cache contract and content fingerprints,
not an independent rerun of the backbone model. Embedding files stay unchanged in shared scratch.

## Loader and batches

```python
from torch.utils.data import DataLoader
from csrv2_repro.training_data import RetrievalTrainingDataset, collate_training_items

train = RetrievalTrainingDataset("/path/printed/by/builder/dataset.json")
validation = RetrievalTrainingDataset("/path/printed/by/builder/dataset.json", "validation")

# Each item has a (task, kind, ID) key, its embedding, and all positive query-group memberships.
item = train[0]
# Illustrative only: the E4/E5 sampling and batch-size decisions are still open.
loader = DataLoader(train, batch_size=16, collate_fn=collate_training_items)
batch = next(iter(loader))
# batch['embeddings']: [B, 2560]; batch['positive_mask']: [B, B], bool, diagonal false.
```

Construction checks the bundle/index/file hashes and manifest selection. Arrays are opened as
read-only memory maps; returned vectors are copies, so preprocessing cannot alter shared caches.
Training iteration yields only training queries plus the complete corpus. `query_vector(task, qid)`
rejects validation/test IDs absent from that view. Validation is a separate view; only NFCorpus and
FiQA currently have validation queries. Its corpus is still the shared retrieval corpus, not unseen
validation documents. `test` is deliberately unsupported by this training loader.

Supervision helpers:

- `query_ids(task)`, `document_ids(task)`, `query_vector(task, qid)`, `document_vector(task, did)`.
- `judgments(task, qid)` returns every original grade, including zero.
- `positive_pairs()` yields `(task, query_id, document_id, grade)` for grades above zero.
- `relevance_matrix(query_keys, document_keys)` preserves grades; `-1` denotes unjudged pairs,
  distinct from explicitly judged nonrelevance (`0`). Keys are `(task, ID)` tuples.
- `positive_mask(query_keys, document_keys)` marks the directly judged positive query/document edges.

For SAE batches, each query defines one group. A document can be in several groups; two different
items are positive if they share a group. Sharing a document does not merge two query groups.
Repeated copies of the same item and diagonal entries are excluded. Unrelated documents have no
supervised group. Task namespaces prevent identical IDs from different datasets becoming positives.
This fixes the scalar-label information loss described in the upstream audit (D-009).

The collator preserves available positives; it does not guarantee that a uniformly sampled batch
contains a positive pair. A sampler that includes meaningful positive pairs, the SAE losses,
hyperparameters and pilot runs remain the next E4/E5 deliverables. Raw grades remain available for
auditing; the membership mask uses positive/nonpositive membership, without introducing grade weights.

## Checks

```bash
PYTHONPATH="$PWD/src" uv run --frozen --no-sync pytest -q tests/test_training_data.py
```

Tests cover nonnumeric lexicographic corpus order, chunk boundaries, query row selection,
held-out exclusion, shared positives, graded/zero judgments, cross-task IDs, nontransitive group
memberships, cache-contract parity, corrupted arrays, and read-only cache protection.
