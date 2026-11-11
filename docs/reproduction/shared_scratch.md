# Shared scratch between scur0933 and scur0944

Set up 2026-11-05 by scur0933 (francesco). Embeddings, the HF cache and per-query predictions live once,
under scur0933's scratch, and both accounts can read and write them. Access is granted with POSIX ACLs
for these two users only; the course-wide Unix groups have no access.

## What is shared
| Path | What | Size |
|------|------|------|
| `/scratch-shared/scur0933/csrv2/emb_cache/` | Embedding cache (`CachedEncoder`), one dir per run name, e.g. `qwen3_4b_dense_prompt_query_fp32/<Task>/{query,document}-<n>-<hash>.npy` | 7.4 GB |
| `/scratch-shared/scur0933/csrv2/hf/` | `HF_HOME`: Qwen3-Embedding-4B, CSRv2-retrieval ckpt, mteb datasets | 8.4 GB |
| `/scratch-shared/scur0933/csrv2/predictions/` | All results (summary.json, mteb outputs, `*_predictions.json`), same layout as `results/`; results are not in git (D-009) | 0.5 GB |

Not shared: `uv-cache/` and `wandb/` stay per user.

## One-time setup for scur0944 (agent: run this on a login node)
`jobs/snellius/env.sh` uses `SCRATCH_ROOT=/scratch-shared/$USER/csrv2`, and the code finds the cache at
`$SCRATCH_ROOT/emb_cache`. Point your own scratch root at the shared dirs with symlinks; no code change needed.

```bash
SHARED=/scratch-shared/scur0933/csrv2
MINE=/scratch-shared/scur0944/csrv2
mkdir -p "$MINE"
for d in emb_cache hf predictions; do
  if [ -e "$MINE/$d" ] && [ ! -L "$MINE/$d" ]; then mv "$MINE/$d" "$MINE/$d.old"; fi   # keep, don't delete
  ln -sfn "$SHARED/$d" "$MINE/$d"
done
ls -l "$MINE"; ls "$MINE/emb_cache"     # should list the run dirs below
```
Delete `*.old` only after checking nothing in them is missing from the shared copy.

## Using it
- **Cached embeddings are reused automatically.** The cache key is model, revision, dtype, max_seq_length,
  padding side, prompt and the exact texts. Same config means a cache hit, so no GPU encoding is needed.
  Example: `uv run python -m csrv2_repro.e2_mrl --config configs/E2_mrl_query.toml --dtype float32` runs on
  cached E1 embeddings. Runs available: `qwen3_4b_dense_prompt_{query,both}_fp32` (8 retrieval tasks),
  `released_ckpt_fp32` (E6b, 8 tasks), and `qwen3_4b_dense_{bf16,fp32}`, `released_ckpt_{bf16,fp32}` (E0/E1p, NFCorpus+SciFact).
- **Results into your repo checkout** (summaries, paired tests): `rsync -a /scratch-shared/scur0933/csrv2/predictions/ results/`.
  They are gitignored, so they never get committed.

## Rules
- Do not delete or overwrite files the other person created. Add new run dirs instead.
- Do not run the same run name from both accounts at the same time: both would write the same `.tmp.npy`.
  Claim the run in `ledger/claims.md` first, as usual.
- New files inherit the ACL automatically. Files moved in with `mv` from elsewhere do not: after a `mv`, run
  `setfacl -R -m u:scur0933:rwX,u:scur0944:rwX,m::rwX <path>`.
- If access breaks: `getfacl /scratch-shared/scur0933/csrv2/emb_cache` should list both users with `rwx`.
  scur0933 must fix it (only the owner can change ACLs on these dirs).
- Scratch is not backed up and is purged after a period of inactivity (check Snellius docs). Anything that
  matters is code and configs in git; copy result dirs you need to keep somewhere backed up.
