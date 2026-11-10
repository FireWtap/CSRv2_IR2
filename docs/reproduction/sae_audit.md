# CSRv2-linear: upstream training audit and reuse

Audited 2026-11-10 against upstream commit `3b2d4d27ec1f9d70af943806ece8dfd665b8d28b`
and [the paper, Eqs. 2–8 and Appendix C.3](https://arxiv.org/html/2602.05735v1).
The fetched LaTeX source confirms the transcribed equations and Qwen hyperparameters;
its hash is recorded with the audit. This audit concerns the SAE trained on frozen
original Qwen3-Embedding-4B vectors, with no LoRA or backbone optimization.

**The training code is available, but the complete trainer cannot be certified as a
correct implementation of the printed paper.** Reuse its verified components directly;
do not launch `CSR_training.py` unchanged and call that the paper reproduction.

## What we can reuse

| Component | Evidence | Our implementation |
|---|---|---|
| Linear encoder, pre-bias, latent bias, tied decoder | Eqs. 2–3; Appendix C.3; `model_zoo.CSR` | Inherit the upstream class and initialization directly |
| Encoder/decoder forward operations, no layer normalization | Upstream constructs `normalize=False`; matches printed sparse transform | Inherit `encode_pre_act`, `decode`, `preprocess`, `csr_forward`, `forward` |
| Nonnegative TopK output | Upstream TopK then ReLU gives the same output as printed ReLU then TopK | Retain the operation; replace bookkeeping/mutation locally |
| Reconstruction objective | Trainer uses MSE(k) + MSE(4k)/8 | Keep weights and upstream mean-MSE convention |
| Linear k schedule | `get_current_k` reaches target at 70% of steps | Call the original function; validate capacity before calling |
| Adam | Appendix C.3 specifies Adam; upstream uses epsilon 6.25e-10 | Same optimizer/epsilon, explicit lr and weight decay |

`src/csrv2_repro/upstream_sae.py` imports the original files from the submodule.
It does not vendor a second encoder or scheduler implementation. Temporary aliases
resolve upstream's bare imports without permanently replacing application modules.
No script is executed with its working directory inside `third_party/`.

The model retains upstream state-dict keys and tied parameter identity. Tests compare
the reused sparse projection, reconstruction, and reconstruction gradients exactly
against upstream initialized with the same weights, and verify save/load parity.
Our `encode` method applies the same projection without changing activity counters.

## Confirmed defects and paper differences

1. **Label key mismatch eliminates the supervised signal.** Retrieval preparation saves
   `label`; `CustomDataset` reads `labels`. It substitutes scalar zero for every missing
   label. The trainer accepts that tensor as valid, so every example becomes positive
   and no negatives remain. A deterministic test shows loss and gradients both equal
   zero. This is more precise than saying the code simply disables the CL branch.
   Our core consumes E3's explicit positive mask, bypassing this loader.

2. **The upstream supervised loss is not the printed Eq. 7.** It normalizes sparse
   features, divides similarities by temperature 0.2, and concatenates two identical
   deterministic forward passes. The duplicate of each example supplies a self-positive,
   even when no distinct labeled positive occurs in its batch. Eq. 7 instead writes dot
   products over supervised positive/negative sets. The log of a *sum* of positive
   exponentials is consistent with the paper; it must not be replaced by the more common
   average of individual positive log-probabilities. Our stable logsumexp implementation
   follows the printed expression and D-009 many-to-many masks. Numerical values and
   gradients are checked against the equation independently.

3. **The auxiliary scaling differs.** Upstream divides auxiliary MSE by the batch
   variance of its residual target. The printed definition is plain residual MSE.
   This changes its effective weight and has a zero-variance corner case. Our paper
   core uses plain MSE, decodes only dead-feature contributions without a pre-bias,
   and preserves the upstream stop-gradient on the primary reconstruction target.
   That stop-gradient and mean reduction are implementation conventions, not fully
   specified by the printed norm notation.

4. **TopK has avoidable side effects.** `auxk_mask_fn` changes the input tensor using
   `.data`, so the returned preactivations are silently masked. `top_k` updates inactivity
   counters on every call, including evaluation; the supervised trainer calls it three
   times per batch (k, 4k, k). It also references an uninitialized local when `auxk=0`.
   The local override preserves the input, supports disabled auxiliary activation,
   and advances counters only on the target-k training pass. The 4k branch does not
   reset activity. The paper does not specify this counter convention; D-011 records it.

5. **Data preparation needs the already implemented E3 corrections.** Scalar document
   labels overwrite shared-query memberships; raw qrel rows lose grades; test-derived
   training queries are later evaluated in the full test set. D-007/D-009/D-010 fix these
   outside upstream and preserve our frozen query manifest. No new embeddings are needed.

6. **The highest-k schedule is inconsistent with the stated capacity.** With h=10240,
   final k=4096 requires 4k=16384 and cannot run in this trainer. Even using final k=2560,
   the printed initial k=4k=10240 would require a 40960-wide initial reconstruction
   branch. Upstream silently caps initial k at h/4, removing annealing at target 2560;
   for target 4096 it raises initial k back to 4096 and still overflows. Our wrapper
   rejects these schedules rather than silently choosing a different experiment.
   Targets 2, 4, 16 and 64 fit with the printed initialization.

7. **CLI defaults are not the Qwen paper recipe.** The trainer defaults to epochs=10,
   batch=4096, lr=4e-4, weight decay=0, and no annealing when initial k is omitted.
   Appendix C.3 explicitly lists Qwen epochs=128, batch=256, lr=2e-5, weight decay=1e-4.
   We found no SAE-specific evidence that epoch and batch values were swapped.
   Backbone `trainer_state.json` cannot settle an SAE training question. D-003 remains
   open; do not silently substitute defaults.

The full trainer also contains distributed-specific hazards (`model.pre_bias` access
through a DDP wrapper, and a contrastive self-mask without the rank offset). Our new
core explicitly supports a single process; these paths are not certified by CPU tests.

**Correction to the earlier review:** the k and 4k reconstruction terms *are* the
paper's multi-TopK objective. A separately summed loss over every evaluation k is
not required by the paper. Their presence can support a shared head evaluated at
different k; this alone does not establish which training target produced each table row.

## Local training core and boundaries

`VerifiedCSR`, `annealed_k`, `linear_loss`, and `make_optimizer` form a reusable,
single-device training core. `linear_loss` accepts `collate_training_items` output
from E3, detaches its frozen input vectors, and returns all objective components.
It rejects duplicate item keys, impossible 4k values, nonfinite input/loss, and a
supervised batch containing no distinct positive pair. No backbone is instantiated.

Missing-positive anchors contribute zero and the sum is divided by the complete
batch size. Under D-012, items with empty query-group memberships are excluded from
both contrastive anchors and candidates: they cannot receive gradients through another
example's negative term. Distinct labeled items can serve as in-batch negatives even
without an in-batch positive; this does not assert their ground-truth irrelevance.
These choices are explicit extensions for incomplete labels, recorded in D-011/D-012.
The paper does not settle them. The five missing-positive ArguAna queries, and unlabeled
corpus documents, contribute to reconstruction/auxiliary losses only.

The second review found and repaired this exclusion bug in the initial D-011 core.
Previously, a zero anchor contribution was mistaken for zero supervised gradient;
an empty-membership example still received gradients as a negative candidate. The
new regression failed on the original local core and passes after the D-012 repair.

This core follows the **printed equations**, with the conventions above. It is not
a claim that those equations were exactly what the authors executed. In particular,
cosine/temperature and residual normalization could be undocumented implementation
choices rather than accidental bugs. An unchanged-code comparison would need its
own named protocol, not silent substitution in this core.

No full trainer/job or sampling distribution has been selected here. The original
preparer repeats query and document vectors for each qrel, then shuffles rows; E3
stores each vector once. A future sampler must explicitly decide pair frequency,
unpaired-document exposure, and how distinct positives share batches. Uniformly
shuffling E3 rows is not equivalent to upstream pair-row frequency. This remains
the next pipeline decision, alongside the full epoch budget and highest-k policy.

## Reproducible checks

Run from the reproduction worktree (Python always through uv):

```bash
PYTHONPATH="$PWD/src" OMP_NUM_THREADS=2 uv run --frozen --no-sync pytest -q tests/test_upstream_sae.py
PYTHONPATH="$PWD/src" uv run --frozen --no-sync python scripts/audit_sae.py \
  --dataset /path/to/E3/dataset.json \
  --output outputs/sae_audit.json
```

The audit script reproduces the label defect and the objective difference, verifies
the E3 artifact checksums, and performs one CPU optimizer step on two actual query/doc
pairs (ArguAna and NFCorpus). It deliberately uses a small 128-wide head for correctness,
not the paper's 10240-wide model. It checks nonzero supervised gradients, finite gradients,
and a weight update. It does not save a trained checkpoint or measure retrieval quality.

The initial recorded result is `results/E5/upstream_audit.json`, with code/upstream commits,
source hashes, input manifest identity, explicit smoke configuration, and SLURM id
(null for this CPU check). Unit tests additionally cover the paper equation and gradients,
auxiliary behavior, annealing, input immutability, tied weights, and checkpoint round-trip.

For the second review, add `--review` to `scripts/audit_sae.py`. It runs 64 CPU check
steps on one real positive pair per task plus the five missing-positive ArguAna queries
(21 distinct items), using a 128-wide head and a 16-to-2 schedule. Every step checks
finite gradients/parameters, a nonzero supervised parameter gradient, and exactly zero
supervised latent gradient for all five reconstruction-only items. This fixed-batch
check is not an epoch sampler, generalization experiment, or paper-sized GPU pilot.
The follow-up record is `results/E5/upstream_review.json`; the original result is
retained as historical evidence, not overwritten.

Additional unit checks independently reconstruct every objective component and all
parameter gradients, use numerical finite differences for contrastive gradients, follow
dead-feature activation and recovery across the threshold, and compare outputs exactly
with the installed Sentence Transformers evaluation module at multiple inference k.
Passing these checks supports the specified local implementation; it cannot prove
equivalence to undocumented author choices or predict reproduced retrieval scores.
