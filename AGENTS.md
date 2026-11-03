# CSRv2 reproduction — agent & contributor guide

IR2 (UvA MSc) reproduction study of **CSRv2: Unlocking Ultra-Sparse Embeddings** (Guo et al., ICLR 2026,
arXiv 2602.05735). Scope (ledger D-001): **Qwen3-Embedding-4B, the 8 MTEB retrieval tasks only**.
- Primary: Table 3, Retrieval column (MRL / CSR / CSRv2-linear / CSRv2 at k = 2, 4, 16, 64, 2560).
- Secondary: Table 2 retrieval-time column, ported to Qwen3-4B.
- Optional: Table 4 ablation, ported to Qwen3-4B retrieval.

Numbers, task list and hyperparameters: `docs/reproduction/targets.md`. Experiment IDs and order:
`docs/reproduction/plan.md`. Do not open the PDF unless those two files are insufficient.

## Session start (do this, in this order, nothing more)
1. `cat ledger/claims.md` and `tail -n 30 ledger/log.md`.
2. Read `ledger/decisions.md` only if your task touches scope, hyperparameters, env or eval protocol.
3. Claim your work in `ledger/claims.md` before editing anything else.

## Hard rules
- **Never modify `third_party/`.** It holds git submodules pinned to upstream commits. Import from it,
  run its scripts, read it; all our code goes in `src/`, `scripts/`, `configs/`, `jobs/`.
  If upstream is broken, work around it in our code and log a `BUG` entry. `git status` must never
  show `third_party/CSRv2` as modified. Bumping the submodule commit requires a decision entry.
- Run upstream scripts from a run dir (`scripts/upstream_rundir.sh <run_id>`), never with CWD inside
  `third_party/`; they write `./ckpt`, `./CSR_results_*` relative to CWD.
- Always run Python through `uv run` (upstream spawns bare `python`, which must resolve to the venv).
- Large artifacts (embeddings, checkpoints, HF cache, run dirs) go in `outputs/` locally or
  `$SCRATCH_ROOT` on Snellius. Both are gitignored. Never commit `*.npz`, `*.pth`, `*.safetensors`.
- Commit results as small files: `results/<EXP_ID>/*.json|csv` with the git commit, upstream commit,
  config and SLURM job id that produced them. A number not in `results/` does not exist.
- Any deviation from the paper (different k, data, hyperparameter, metric) needs a `D-NNN` entry.

## Ledger (`ledger/`) — the shared memory between people and agents
| File | Purpose | How to write | How to read |
|------|---------|--------------|-------------|
| `claims.md` | who is doing what right now | add a row before starting, delete it when done | `cat` (always small) |
| `log.md` | what happened: DONE / RESULT / BUG / BLOCKED / NOTE | append ONE line, never edit old lines | `tail -n 30`, `grep -n <EXP_ID>` |
| `decisions.md` | durable choices and paper deviations, `D-NNN` | append a new entry; supersede, never rewrite | only when relevant, `grep -n D-0` |

Token discipline:
- One line per log entry; put details in the file the pointer names, not in the ledger.
- Never paste tables, logs, stack traces or diffs into the ledger. Point to `results/`, a commit or a job id.
- Never read the whole `log.md`; tail or grep it. Do not re-read files the ledger already summarises.
- When `log.md` passes ~300 lines, move everything but the last 50 lines to `ledger/archive/log-<date>.md`
  in a dedicated commit.
- `log.md` and `decisions.md` use git `merge=union`, so concurrent appends never conflict.
  `claims.md` does conflict on purpose: resolve it by hand, since a conflict there means two people claimed the same thing.

## Layout
```
AGENTS.md / CLAUDE.md     this guide (CLAUDE.md just imports it)
ledger/                   claims, log, decisions
docs/reproduction/        targets.md (paper numbers), plan.md (experiments E0..E8)
docs/paper/               paper PDF + LaTeX (gitignored; scripts/fetch_paper.sh)
src/csrv2_repro/          our Python package (uv-installed, importable as csrv2_repro)
configs/                  one config per experiment, named <EXP_ID>_<desc>.yaml
scripts/                  small CLI helpers (fetch_paper.sh, upstream_rundir.sh)
jobs/snellius/            env.sh, setup_env.sh, template.job, <EXP_ID>_*.job
results/<EXP_ID>/         committed metrics (small)
outputs/                  gitignored: embeddings, ckpts, run dirs, slurm logs
third_party/CSRv2         upstream submodule (Y-Research-SBU/CSRv2). Read-only.
```

## Environment
- Local (macOS, CPU, dev/debug only): `uv sync`, then `uv run ...`. Python 3.11, versions pinned to
  upstream `requirements.txt` in `pyproject.toml`; `uv.lock` is committed. Change deps only with
  `uv add/remove` plus a decision entry, and commit `pyproject.toml` and `uv.lock` together.
- Snellius: once per account, `bash jobs/snellius/setup_env.sh` on a login node. Every job sources
  `jobs/snellius/env.sh` (module 2024, HF/uv/wandb caches on `/scratch-shared/$USER/csrv2`, wandb offline).
  Copy `jobs/snellius/template.job` to `jobs/snellius/<EXP_ID>_<desc>.job`. Use `--frozen` on the cluster.
- Fresh clone: `git clone --recurse-submodules <repo>` (or `git submodule update --init`).

## Git
- `main` is protected by convention: work on `<owner>/<EXP_ID>-<desc>` branches, open a PR, squash-merge.
  Small ledger-only commits may go directly to `main`.
- Commit messages: `<EXP_ID|area>: <imperative summary>` (e.g. `E2: add MRL truncation eval`).
- Pull/rebase before appending to the ledger; commit ledger updates together with the work they describe.
- Never commit secrets (HF/wandb tokens): use env vars or `~/.cache`.
