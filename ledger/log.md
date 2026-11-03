# Log — append-only, one line per entry, newest at the bottom

Format: `YYYY-MM-DD | owner | exp/area | TYPE | what happened (<=1 line) | pointer`
TYPE in {DONE, RESULT, BUG, BLOCKED, NOTE}. Pointer = path, commit, job id or URL. Never edit old lines.
Read with `tail -n 30 ledger/log.md`, search with `grep -n "E4" ledger/log.md`.

2026-11-03 | francesco/claude | setup | DONE | repo scaffold: AGENTS.md, ledger, uv env (py3.11, upstream pins), CSRv2 submodule @3b2d4d2 | AGENTS.md
2026-11-03 | francesco/claude | setup | NOTE | paper Table 2 & 4 use e5-Mistral-7B; Table 3 is Qwen3-4B; scope set by D-001 | ledger/decisions.md
2026-11-03 | francesco/claude | upstream | BUG | text/data_preparation/get_embeddings_of_each_dataset.py does not parse (IndentationError l.112: docstring/body indent mismatch); we must reimplement it in src/ (E3) | third_party/CSRv2@3b2d4d2
2026-11-03 | francesco/claude | upstream | NOTE | upstream scripts read ./*.json and write ./ckpt ./CSR_results_* relative to CWD and spawn bare `python`: run them via scripts/upstream_rundir.sh under `uv run` | scripts/upstream_rundir.sh
2026-11-03 | francesco/claude | setup | NOTE | released ckpt Y-Research-Group/CSRv2-retrieval = Qwen3-4B + LoRA + SAE(h=10240), k adjustable -> E0/E6b | docs/reproduction/plan.md
2026-11-03 | francesco/claude | E1 | NOTE | precision issue: upstream loads Qwen3-4B without torch_dtype -> fp32 (transformers 4.55 default); fp32 E1 ~7-10 A100-h/protocol vs ~1h bf16; bf16 may flip TopK active sets at small k. Blocking E1 full runs until E1p measures the gap | ledger/decisions.md D-005
2026-11-03 | francesco/claude | E0 | BUG | ST 5.0 SparseEncoder copies SAE k into model.max_active_dims at load and passes it to forward, overriding sae.k: must set both to change k (fixed in our E0 code) | src/csrv2_repro/e0_released_ckpt.py
2026-11-03 | francesco/claude | E1 | NOTE | padding_side matters for absolute-position models (MiniLM NFCorpus: left 29.44 vs right 31.59 = plain mteb); Qwen3 uses RoPE so left (upstream) should be invariant; our CachedEncoder matches plain mteb exactly | src/csrv2_repro/encoders.py
2026-11-03 | francesco/claude | E0,E1 | DONE | E0/E1 code + configs; CachedEncoder == plain mteb (MiniLM NFCorpus 31.59, right pad); E0 cached path == upstream SparseEncoder path (diff 0.00, fp32+bf16, fake ckpt) | src/csrv2_repro/
2026-11-03 | francesco/claude | E1p | NOTE | job ready: bf16 vs fp32 on NFCorpus+SciFact (dense + released ckpt, all k, active-set Jaccard); not yet run on Snellius | jobs/snellius/E1p_precision.job
2026-11-03 | francesco/claude | E1 | BLOCKED | full 8-task E1 waits for D-005 (dtype) from E1p results | configs/E1_dense_*.toml
2026-11-03 | francesco/claude | E1p | NOTE | env set up on Snellius, models+NFCorpus/SciFact pre-downloaded; E1p submitted on gpu_a100, pending (all partitions down) | slurm 28872577
2026-11-03 | francesco/claude | upstream | BUG | test contamination: for 5/8 retrieval tasks (no train split) upstream LoRA finetune + SAE label-CL train on 80% of TEST qrels, eval on full test; affects CSRv2(-linear) rows of Tab.3 -> needs a decision | docs/reproduction/upstream_training.md
2026-11-03 | francesco/claude | upstream | NOTE | released CSR_training.py trains one --topk target per run (no multi-k loss), but model card claims one Multi-TopK SAE for all k; unresolved | docs/reproduction/upstream_training.md
2026-11-03 | francesco/claude | E6b | NOTE | released ckpt trainer_state: LoRA 1 epoch (stopped 16000/17289), per-device bs 4, peak lr 1e-5, warmup ~1800 steps; paper says 10 ep / 2e-5 / 1000 | docs/reproduction/upstream_training.md
2026-11-03 | francesco/claude | upstream | NOTE | leakage check: certain in released code; released ckpt step count (<=69156 pairs) fits recipe with test-derived pairs (61806) not train-only (35085), but not exact | docs/reproduction/upstream_training.md
2026-11-03 | francesco/claude | E1p | NOTE | E1p now saves per-query rankings (top 100) and the report adds paired per-query stats + TOST equivalence (proposed delta 0.2 nDCG pts, to confirm in D-005) | src/csrv2_repro/paired.py
2026-11-03 | francesco/claude | E1,E2,E6b | NOTE | 8-task eval job ready (E1 both+query, E2 MRL, E6b released ckpt all k); dtype passed at submit (DTYPE=..., refuses without); E1->E2 cache path tested on CPU (MiniLM SciFact: E2@full dim == E1, 64.51) | jobs/snellius/eval8_E1_E2_E6b.job
