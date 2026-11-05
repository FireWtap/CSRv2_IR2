# Experiment plan

Planning revision: 2026-11-04. Scope remains D-001: Qwen3-Embedding-4B, the eight retrieval
tasks in `targets.md`, primary Table 3 grid k = {2, 4, 16, 64, 2560}.
Dataset splitting and matched held-out evaluation are accepted in D-007 and implemented in
`docs/reproduction/splits.md`. Other protocol proposals below still require resolution before
affected runs. D-006 uses the released checkpoint for the CSRv2 column; own E6 retraining is optional.

## Evidence reviewed and work already available

- Local review: upstream `3b2d4d27ec1f9d70af943806ece8dfd665b8d28b`; label/labels mismatch
  confirmed with a CPU probe; syntax error and missing TopK finetuning arguments confirmed.
- Collaborator branch reviewed at `01f364f`, `origin/francesco/E0-E1-eval`: training audit,
  E0/E1/E2/E6b evaluators, embedding cache, E1p paired precision report, TOML configs and jobs.
  Code and results are now integrated via main commit `b16f143` (collaborator tip `7567b4c`).
  Preserve the collaborator's active claim.
  Review source: [training audit](https://github.com/FireWtap/CSRv2_IR2/blob/01f364f/docs/reproduction/upstream_training.md).
- SLURM accounting on 2026-11-04 reports E1p job 28872577 COMPLETED, exit 0:0.
  Results are now available in `results/E0/` and `results/E1p/`, including the precision report
  and per-query CSVs. The review and recommended next actions are recorded below.
- Independently checked released model metadata at revision
  `af16198ddd83dec2d747756a1fb592378c15c4ac`: trainer_state records one planned epoch,
  global_step 16000, epoch about 0.925, max_steps 17289 and train_batch_size 4.
  Model metadata records sentence-transformers 5.2.0; adapter metadata records peft 0.18.1.
  Keep the existing locked environment until a demonstrated compatibility issue warrants a decision.

Corrections to carry into the collaborator audit:

1. Its pipeline description says the released finetuning script implements a multi-TopK loss.
   The inspected script does not implement that objective or accept the documented TopK flags.
2. The SAE does use two reconstruction scales, k and 4k. This does not establish that one released
   head reproduces separately trained targets across the full Table 3 grid.
3. Test-label reuse is established in the released preparation/evaluation paths. Whether the
   released checkpoint used those paths remains unproven. Step counts alone cannot identify
   training data without effective batch size, world size, accumulation and dataset revisions.
4. Backbone trainer_state cannot settle D-003's SAE epoch count. The released checkpoint is a
   reference comparator, not an upper bound on what retraining can achieve.

## Questions and proposed protocols

### Precision review and recommended next actions (2026-11-04)

Source: `results/E1p/precision_report.json`. Units below are nDCG@10 points on the 0-100 scale;
differences are bf16 minus fp32, equally weighted over NFCorpus and SciFact only.

| Representation | Mean paired difference | 90% bootstrap CI | CI inside proposed +/-0.2 margin? |
|---|---:|---|---|
| Dense | -0.0974 | [-0.2338, 0.0112] | No |
| Released CSRv2 k=2 | -0.0541 | [-0.1569, 0.0495] | Yes, macro only |
| Released CSRv2 k=4 | +0.1805 | [-0.0813, 0.5071] | No |
| Released CSRv2 k=16 | -0.1513 | [-0.4561, 0.1259] | No |
| Released CSRv2 k=64 | +0.0134 | [-0.1876, 0.2230] | No |
| Released CSRv2 k=2560 | -0.0115 | [-0.1800, 0.1551] | Yes, macro only |

Failure to establish equivalence is not proof that bf16 is materially worse. Differences vary in
sign, and passing a two-task macro check does not establish per-task or eight-task equivalence.
The CSV query counts and means agree with the report, with no duplicate query keys. Unrounded MTEB
nDCG values match between direct/cached paths at k=2 for both tasks and both precisions. Higher-k
direct/cached agreement has not been established by this report. Active-set overlap is a pre-ReLU
diagnostic using an fp32 SAE, not proof of identical end-to-end rankings. Source provenance says
`git_dirty=true`; preserve the run's exact code or clarify whether dirtiness came only from artifacts.

Recommended resolution of D-005: fp32 inference for primary E1/E2/E6b and main SAE comparisons;
bf16 for development and explicitly labeled exploratory runs. Keep inference precision separate from
the future backbone-training precision decision. Do not widen the margin after seeing these results
or spend the next work cycle trying to make the precision test pass. If compute limits require a bf16
main study, record that deviation and retain the measured sensitivity instead of claiming equivalence.

Next priority: complete the eight-task upstream-prompt E1/E2/E6b table while building E3/E4/E5.
Keep query-only prompting as a separate sensitivity experiment. The existing eval8 job runs both
prompt protocols; split or parameterize it before submission if only the primary protocol is intended.
Reuse existing caches only after checking their identity; run-name-specific paths currently prevent
automatic reuse of all E0/E1p caches by full-run configs.

### Proposed ownership and handoffs

Names follow the repository branches: user = Samu; collaborator = Francesco. The user confirmed
this work split; active claims track what each contributor is currently editing.

| Owner | Immediate responsibility | Deliverable | Later responsibility |
|---|---|---|---|
| Francesco | E0/E1p closure and E1/E2/E6b evaluation | Eight-task per-method/per-k results, fixed evaluator and documented precision/prompt settings | E7 timing, result aggregation, independent evaluation of trained heads |
| Samu | E3 training manifests and E4/E5 SAE implementation | Audited train/validation data plus CSR/CSRv2-linear pilots at k=2 and 16 | Remaining SAE grid/seeds, E8 component ablations; lead E6 implementation if feasible |
| Both | Protocol decisions, review of each other's changes, author questions | Accepted decision entries, reviewed manifests, agreed compute budget | E6 go/no-go, interpretation and report writing |

Francesco owns existing `common.py`, `encoders.py`, `paired.py`, `precision_report.py`,
`e0_released_ckpt.py`, `e1_dense_backbone.py`, `e2_mrl.py`, their E0/E1/E1p/E2/E6b configs/jobs
and results. Samu should use new data/SAE modules, E3/E4/E5 configs/jobs/results and focused tests.
The user-authorized D-007 implementation adds minimal manifest hooks to `common.py` and the three
evaluation entry points; new split preparation lives in separate modules. Coordinate these hooks
when merging Francesco's ongoing work. His claim remains intact; no collaborator message was sent.

Agree these contracts first:

1. Frozen run configuration: exact model/tokenizer revisions, dataset revisions, prompts, text
   construction, normalization, length and dtype. Training extraction reuses reviewed preprocessing.
2. Training manifest: dataset/split/query/document IDs, judged relevance, source revisions and
   validation assignment, with overlap checks and a checksum. Preserve many-to-many positives.
3. SAE checkpoint: weights plus input/latent dimensions, trained k, intended inference k,
   preprocessing, seed, train-manifest checksum and full training config.
4. Evaluation handoff: Francesco loads the head onto the matching backbone cache and returns
   per-task/per-query metrics. A small fixed embedding batch must produce the same sparse vectors
   in Samu's training/export code and Francesco's evaluator before any full evaluation.

First parallel work cycle:

- Francesco: finish the evaluator checks (full-dimensional MRL identity; a small actual-checkpoint
  direct/cached check at k=16/2560; effective config and revision provenance), then run primary
  fp32 E1 -> E2 and E6b. Confirm protocol decisions before launching; measure actual runtime.
- Samu: build the train/validation manifest and overlap audit, fix the label contract in our own
  modules, and demonstrate meaningful supervised gradients and save/load parity on tiny inputs.
  Code and audit can proceed while evaluation runs; real-data embedding extraction and full training
  follow the agreed protocol. Resolve D-003 before full training; label short pilots as smoke runs.
- Joint checkpoint: review the eight-task table and the data/SAE smoke evidence. Then expand E4/E5
  to the full k grid, with repeated seeds at k=2 and 16. Decide E6 only after measuring pilot cost
  and resolving or explicitly reconstructing the missing TopK-LoRA recipe.

Training with the protected query partitions in D-007 is the accepted protocol below. Keep the released-checkpoint
evaluation distinct, since its training-data provenance is unresolved. The deadline and GPU budget
are still unknown; milestones above specify ordering and handoffs without promising calendar dates.

Report two distinct outcomes:

- **Released-checkpoint reproduction:** run the authors' checkpoint on the original eight test
  sets using a documented evaluation protocol. Compare with Table 3, while stating that its
  training provenance is unresolved.
- **Retraining with protected test queries:** train on the D-007 manifest and evaluate all methods
  on its held-out query IDs against the full corpus. Official train/dev/test partitions are retained;
  test-only tasks use a shared seeded 80/20 query split. This differs from the paper's full-test
  protocol, so numerical agreement with Table 3 is not assumed. Released-checkpoint training overlap
  remains unknown even when evaluated on these IDs.

The frozen supervised pool includes all eight tasks. See `results/E3/split_manifest.json` and
its audit for exact assignments, pinned revisions and zero query-ID overlap. Optional validation
must come from the training pool before sampling/capping; existing dev splits are preserved.
The default does not synthesize extra validation queries. LoRA and SAE must share this manifest;
do not add held-out query texts back as unsupervised singleton rows. Full-corpus unlabeled exposure
is transductive, and document disjointness is not required. See `splits.md` for commands and API.

Use the upstream MTEB prompt protocol as the proposed primary reproduction protocol and query-only
instructions as a separately reported sensitivity analysis. Do not choose a protocol solely because
its test average is closest to 58.37. Any comparison across methods must use matched preprocessing.
The collaborator's D-004/D-005 are now in the local ledger; reconcile the proposals above with
those entries before recording a final protocol decision.

## Experiment inventory from the integrated branch

Configs use `configs/<ID>_*.toml`.

| ID  | What                                                                 | Paper ref        | Depends on | Status  |
|-----|----------------------------------------------------------------------|------------------|------------|---------|
| E0  | Released `CSRv2-retrieval` ckpt on NFCorpus+SciFact, all k, fp32+bf16; upstream path == cached path check | Tab.3 (per-task n/a) | env | results available; protocol review pending |
| E1p | Precision check bf16 vs fp32: dense Qwen3-4B + E0 ckpt on NFCorpus+SciFact -> results/E1p/precision_report.json | D-005 | env | results available; protocol review pending |
| E1  | Dense backbone, 8 tasks, prompt protocol both + query, cache embeddings | Tab.3 row 1      | E1p (dtype) | ready (job eval8, needs DTYPE from D-005) |
| E2  | MRL: truncate cached E1 embeddings to k                               | Tab.3 MRL        | E1         | ready (job eval8) |
| E3  | Frozen query partitions, then training embeddings (replaces broken upstream preparation) | App. B | env | D-007 manifest/export/eval integration done; embedding extraction todo |
| E4  | CSR (no anneal, unsupervised CL) on E3, eval on cached E1            | Tab.3 CSR        | E1, E3     | todo    |
| E5  | CSRv2-linear (anneal + SpSCL) on E3                                   | Tab.3 CSRv2-lin  | E1, E3     | todo    |
| E6  | CSRv2: TopK-LoRA backbone finetune, re-embed, train SAE, eval         | Tab.3 CSRv2      | E3         | optional (D-006) |
| E6b | Released `CSRv2-retrieval` ckpt, all 8 tasks x all k                  | Tab.3 CSRv2      | E0         | ready (job eval8) |
| E7  | Retrieval time, 1M-doc DB, relative to CSRv2@k=2                       | Tab.2 time col   | E2, E5/E6  | todo    |
| E8  | Ablation: CSR, +anneal, +supervise, CSRv2-linear, CSRv2 x k grid      | Tab.4 (ported)   | E4-E6      | optional|

## Stages and completion gates

| Stage / IDs | Work | Required evidence before proceeding |
|---|---|---|
| 0: integrate and audit | Reuse collaborator evaluation code; obtain E1p artifacts; reconcile both reviews | Reviewed code commit, result provenance, no duplicate precision job |
| 1: E0 + E1p | Released checkpoint and dense model on NFCorpus/SciFact; direct vs cached evaluation; fp32 vs bf16 | Valid per-query metrics, matching cache identities, documented precision decision |
| 2: E1 + E2 + E6b | Eight-task dense, native MRL and released checkpoint, all five k values | Complete per-task scores and equal-weight eight-task means; protocol variants clearly identified |
| 3: E3 | Freeze supervised train/validation manifests and produce training embeddings | No held-out query/qrel overlap, correct label schema, reproducible sampling and revisions |
| 4: E4 + E5 | Train CSR and CSRv2-linear on identical frozen embeddings/data | Pilot at k=2 and 16; useful supervision and finite gradients; then full k grid |
| 5: E6 | Implement documented TopK-LoRA objective, re-embed and train SAE | Training recipe resolved or explicitly declared reconstruction; pilot fits compute budget |
| 6: E7 / E8 | Sparse retrieval timing and component ablations | Quality reproduction complete; remaining compute sufficient |

### Stage 0-1: trustworthy evaluation

Review the now-integrated `results/E1p/precision_report.json`, source E0/E1p summaries and
per-query results, checking the exact code/configs used by job 28872577. If that job predates
the paired-statistics code, derive the report from saved predictions or evaluate existing caches;
do not repeat model inference unless necessary.

Before accepting the evaluation implementation:

- Verify cached vs direct evaluation on the actual released checkpoint, including k=2, 16, 2560.
  Set both SAE k and SparseEncoder.max_active_dims; check actual nonzero counts after ReLU.
- Check MRL at k=2560 exactly matches its corresponding dense cache. The paper's small difference
  between full-dimensional MRL and backbone is an open protocol question, not a target to force.
- Retain graded qrels, MTEB's identical-ID policy, query/document text construction and cosine scoring.
  Recompute per-query scores and check their aggregate against MTEB; record zero-score tie handling.
- Validate padding/batch changes empirically. Store full precision scores; round only for presentation.
- Pair precision caches by content/ID manifests, not only task/side/row count as the current report does.
  Require identical query sets and fail on missing result files.
- Keep post-ReLU support agreement distinct from the report's pre-ReLU TopK overlap. Its fp32-SAE
  diagnostic isolates backbone precision effects; it does not test all end-to-end precision effects.
- Decide the practical precision margin before inspecting comparisons. The collaborator proposes
  0.2 nDCG points (0-100 scale). Treat this as pending; report per-task/per-k differences and paired
  confidence intervals, not only the macro mean. Two-task equivalence does not prove all-task equivalence.

Cache fingerprints/provenance should include actual base-model and adapter revisions, tokenizer,
dataset revisions/IDs, preprocessing, prompts, dtype, maximum length and code/config hash.
Save effective CLI overrides, not just the unmodified config. Reusing MTEB output must not silently
reuse scores from a different configuration. Pin the underlying Qwen revision as well as the adapter.

### Stage 2: checkpoint and baseline results

Produce the eight-task Table 3 subset for dense, MRL and the released checkpoint before training.
E2 reuses E1 embeddings; E6b needs its own LoRA backbone cache. Report every task and all five k values,
plus task-macro means, paper deltas and paired uncertainty where comparisons share queries.
If results diverge, inspect preprocessing/precision/checkpoint identity using the fixed diagnostic
protocol; report alternative protocols separately rather than tuning on test scores.

### Stage 3-4: controlled SAE reproduction

Resolve the following before full training:

- Preserve label information through NPZ writing/loading; fail on missing labels in supervised runs.
- Preserve many-to-many query/document positives. The upstream scalar document-label assignment can
  overwrite membership when multiple queries share a relevant document. Document the chosen positive
  mask and sampling rule; test that real positive pairs occur in batches and contribute gradients.
- Implement repairs in `src/`, never `third_party/`. Test loss components, k schedule, save/load parity
  and train/evaluation transforms with small deterministic inputs before GPU pilots.
- Resolve D-003 from SAE-specific evidence or an author response. A short smoke run may use fewer
  steps but is not the paper result. Do not silently replace the published 128 epochs with defaults.
- Resolve Table 7's Qwen k=4096 versus Table 3's 2560 and latent capacity/4k constraints.
  Record training k separately from evaluation k. Propose separately trained SAE targets as the
  transparent reconstruction if the original recipe remains unavailable; distinguish E6b's shared head.
- Keep data, optimizer budget and evaluation fixed between CSR and CSRv2-linear. Pilot at k=2 and 16,
  then run the full grid. Plan three seeds for these anchor k values and report seed variation separately
  from query-bootstrap uncertainty; expand to all k if the measured budget permits.

### Stage 5: full CSRv2

Obtain the actual TopK finetuning implementation or implement the paper objective with a documented
decision. Verify gradients across the specified k values and specify physical batch size, accumulation,
cross-device negatives and effective update batch separately. Gradient accumulation does not itself
increase the in-batch negative pool. Avoid the upstream distributed override that forces accumulation=1.

Select checkpoints on reserved validation data. After finetuning, regenerate embeddings and retrain
the SAE; E1 caches belong to the unfine-tuned backbone and cannot substitute. Measure pilot runtime,
memory and storage before committing the full allocation. If budget allows, add a matched ordinary-LoRA
control to distinguish task adaptation gains from TopK-objective gains. Retain the distinction between
reconstructing the paper recipe and reproducing the released checkpoint's recorded training setup.

### Stage 6: timing and ablations

E7 is a Qwen port under D-001. Freeze corpus/query composition, document count, hardware, threads,
dtype and top-k retrieval policy. An eight-task corpus may be smaller than the paper's million-document
retrieval-plus-reranking database: specify any extra data/sampling as a decision; do not silently duplicate
documents. Use an actual sparse index and verify rankings against exact cosine search. Measure index
construction, query encoding, retrieval latency/throughput and memory separately, including index/value
storage. Warm up and repeat timing; dense matrix multiplication on zero-filled vectors is insufficient.
E8 isolates annealing, supervision and finetuning with matched data/budgets; prioritize k=2 and 16.

## Scope, budget and deliverables

First milestone: E0/E1p + E1/E2/E6b and a protocol/provenance audit.
Core retraining milestone: E3/E4/E5. D-006 fills the CSRv2 reference column with E6b;
E6 is optional. Keep full-test artifact results separate from D-007 held-out comparisons,
and state which training stages were actually reproduced.
E7/E8 remain secondary/optional. Estimate GPU hours from pilots, not unverified throughput claims.
Project deadline and remaining GPU-hour allocation are still needed to schedule stages 4-6.

Each completed experiment needs small JSON/CSV outputs under `results/<ID>/`, per-task scores,
code/upstream commits, effective configs, dataset/model revisions, seeds and SLURM IDs. Large caches,
predictions and checkpoints stay in `outputs/` or scratch; retain compact per-query score files where useful.
No task is complete based solely on a submitted/completed job or an uncommitted log claim.

Author questions to draft (not yet sent): exact train/validation/test query manifests; SAE label schema;
actual TopK finetuning code; Qwen SAE epochs and k schedule; one head per k vs one shared head; and
which evaluation protocol/checkpoint revisions generated Table 3.
