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
