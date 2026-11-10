# Decisions — append-only, numbered. Supersede with a new entry; never rewrite an old one.

Format:
## D-NNN (YYYY-MM-DD, who) Title
Decision. Why (1-2 lines). Status: active | superseded by D-MMM.

## D-001 (2026-11-03, francesco) Scope: Qwen3-Embedding-4B, retrieval only
Primary target is Table 3 (Qwen3-4B) Retrieval column; Table 2's retrieval-time column and Table 4's
ablation are ported to Qwen3-4B + retrieval as documented deviations (paper ran them on e5-Mistral-7B).
Why: Qwen3-4B has native MRL (no MRL finetune needed) and a released retrieval checkpoint; e5-Mistral-7B
needs batch-2048 LoRA MRL training. Status: active.

## D-002 (2026-11-03, francesco) Compute and environment
GPU work runs on Snellius (SLURM, jobs/snellius/). Env = uv + pyproject.toml + committed uv.lock, Python 3.11,
pins copied from upstream requirements.txt for the packages the text pipeline imports. Caches on
/scratch-shared/$USER/csrv2 (jobs/snellius/env.sh). Status: active.

## D-003 (2026-11-03, open) Qwen3-4B SAE hyperparameters
Paper Table 7 lists epoch=128, batch=256 for Qwen3-4B (e5: epoch=10, batch=128). 128 epochs is
suspicious. Default to the upstream script defaults/README command until checked against the released
ckpt's trainer_state.json. Status: open — resolve before E4/E5.

## D-004 (2026-11-03, francesco) Eval protocol details for Qwen3-4B
Follow upstream eval_all(): instruction prompts from upstream MTEB_task_prompts.json, left padding, max_seq_length 512,
cosine similarity, mteb 1.38.39 test split. Upstream applies the task instruction to queries AND documents
(model.prompts keyed by task name); Qwen3 recommends queries only. E1 runs both protocols to find which one
reproduces 58.37; the winner becomes the protocol for all later experiments. Batch size 16 instead of upstream's 2
(padding only; invariant for RoPE + last-token pooling). Status: active (protocol choice pending E1).

## D-005 (2026-11-03, francesco) Inference precision: fp32 (upstream) vs bf16
Upstream inference is fp32 by default. fp32 E1 costs ~7-10 A100-h per protocol; bf16 ~1h. Before any full run,
E1p measures the bf16-fp32 gap on NFCorpus+SciFact for (a) dense Qwen3-4B nDCG@10 and (b) the released CSRv2
ckpt nDCG@10 and TopK active-set agreement at k in {2,4,16,64,2560}. Status: open — decide after E1p.

## D-006 (2026-11-04, francesco) Table 3 CSRv2 column = released checkpoint
The CSRv2 column of Table 3 is filled with the released `Y-Research-Group/CSRv2-retrieval` ckpt (rev af16198, E6b),
not with our own retraining. Principle: use the authors' artifacts wherever they exist, and flag any gap between them
and the paper (known so far: ckpt trainer_state differs from the paper recipe, and test-qrel leakage in its recipe;
see docs/reproduction/upstream_training.md). Own retraining (E6) stays optional. CSR and CSRv2-linear have no released
Qwen3-4B ckpt (HF org checked 2026-11-04), so those columns need E3-E5. Status: active.

## D-007 (2026-11-04, Samu / codex) Shared query-held-out training and evaluation
User authorized following the missing-train fallback while evaluating only actual held-out queries.
Preserve official train/dev/test; for the five test-only tasks split sorted query IDs 80/20 with
NumPy RandomState(seed=42), rounding the held-out count up. All judgments for a query stay together.
LoRA and SAE preparation must consume the same frozen manifest; never add held-out query texts or
labels back as singleton training examples. Optional validation comes only from the training pool;
default fraction is 0 (keep the requested 80/20), while existing dev splits remain validation.
Search the full corpus and compare all methods on identical manifest test queries. Full-corpus
unlabeled exposure is transductive; query-disjointness does not guarantee distinct document/text content.
This fixes the released recipe's full-test reuse and intentionally replaces its inconsistent row/query
splits with one deterministic query partition. Held-out scores are a separate protocol, not directly
comparable to paper Table 3 averages. D-006's full-test released-checkpoint reference remains separate;
its training overlap stays unknown even when rescored on our held-out subset. Pin MTEB 1.38.39 dataset
revisions and verify qrel hashes in our loader; upstream remains read-only. Details and commands:
docs/reproduction/splits.md; manifest: results/E3/split_manifest.json. Status: active.

## D-008 (2026-11-05, Samu / codex) Complete the existing dense fp32 query cache
User authorized encoding the missing official train/validation queries for NFCorpus, FiQA2018 and
SciFact. Match the populated E1 cache: pinned Qwen3-Embedding-4B revision 5cf2132, fp32 inference,
left padding, max length 512, batch 16 and upstream task instructions on queries. Query prompts are
identical for E1 both/query modes; encode once, preserve ordered IDs and D-007 manifest provenance.
This completes data extraction for the existing fp32 protocol; it does not settle the broader
D-004/D-005 sensitivity comparisons or authorize training. Status: active.

## D-009 (2026-11-05, Samu / codex) E3 indexed training pool and many-to-many supervision
Complete the user-authorized dataset builder using D-007 query partitions and the existing D-008
dense fp32 cache. The reconstruction pool contains each selected query once and every corpus document
once per task (explicitly transductive). Validation/test queries are absent from the training view;
only training qrels create its supervision. Preserve original grades, with grade > 0 defining positive
query-group membership. A document keeps all its positive query groups; do not overwrite one scalar
label or merge groups transitively. Collation marks distinct items sharing a positive query group;
unjudged/zero-grade relations create no positive membership. Group IDs include the task namespace.
Match the actual MTEB 1.38.39 corpus order (descending string IDs, 50000-row chunks), despite its
misleading length-sort log message. Verify pinned text/model/prompt cache keys and array checksums
before reuse. Epoch sampling and SAE loss implementation remain E4/E5 work. Status: active.

## D-010 (2026-11-10, codex) Preserve queries whose judged documents are absent
The pinned ArguAna corpus is missing the positive documents for five D-007 training queries.
Keep their original judgments and query embeddings in the indexed dataset, and explicitly audit the
missing document IDs. They remain reconstruction examples but have no available supervised positives;
do not fabricate vectors, silently delete judgments, or change the frozen query split. Positive-pair
iteration only emits documents present in the corpus. These queries have empty collator memberships.
The source-coverage check found no such gaps in the other seven tasks or held-out/validation splits.
This refines D-009's handling of incomplete source data; final counts are recorded in the E3 audit.
Status: active.

## D-011 (2026-11-10, Samu / codex) Reuse verified upstream SAE components; distinguish paper from code
User requests CSRv2-linear on frozen original Qwen vectors and maximal upstream reuse after verifying
paper agreement. Import the pinned upstream CSR class and k scheduler directly. Keep its initialization,
tied weights, biases, preprocessing/forward/decoder methods, mean-MSE convention and Adam epsilon.
Repair TopK locally: preserve preactivations, support auxk=0, count dead-feature inactivity once per
target-k training pass and never during inference (threshold 30 and activity cutoff 1e-5 remain upstream).
For the paper-equation core, implement Eq. 7's dot-product log-sum-positive ratio using D-009 memberships,
without upstream's cosine/temperature-0.2 transform or duplicated self-positive views; use ordinary
auxiliary residual MSE without upstream's variance normalization. Preserve a detached reconstruction
target for auxiliary gradients; remove decoder bias from the auxiliary prediction. Unpaired anchors
contribute zero, averaging over the entire batch; reject batches with no positive pair and duplicate item
keys. Other distinct in-batch items act as negatives, not verified nonrelevant documents. These explicit
choices address underspecified edge cases; they are not proof of the authors' historical implementation.
This single-process core changes released-code behavior where it disagrees with the printed recipe;
do not report it as an unchanged upstream run. No LoRA, full-run sampler, epoch budget or highest-k
workaround is selected by this decision; CPU checks use a small head only for correctness. D-003 remains
open for the full-run recipe. Audit and executable checks: docs/reproduction/sae_audit.md. Status: active.

## D-012 (2026-11-10, codex) Enforce reconstruction-only exclusion from contrastive candidates
Second review found that D-011's zero contribution for an unpaired anchor did not prevent gradients
through that item as another anchor's negative. This contradicted D-010's reconstruction-only promise.
Require E3 per-item query-group memberships in supervised batches; items with empty membership are
excluded from both contrastive anchors and candidates, including the five missing-positive ArguAna
queries and unlabeled corpus documents. They still contribute to reconstruction and auxiliary losses.
Other labeled items without an in-batch positive can remain negatives; the contrastive sum is still
divided by the full batch size under D-011. This supersedes only D-011's treatment of empty-membership
items as negatives. This is our explicit incomplete-label convention, not an author-verified recipe.
Regression, finite-difference, independent objective-gradient, dead-feature lifecycle and evaluation
parity checks accompany the repair. Status: active.
