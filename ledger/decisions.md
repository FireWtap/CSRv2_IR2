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
