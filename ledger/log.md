# Log — append-only, one line per entry, newest at the bottom

Format: `YYYY-MM-DD | owner | exp/area | TYPE | what happened (<=1 line) | pointer`
TYPE in {DONE, RESULT, BUG, BLOCKED, NOTE}. Pointer = path, commit, job id or URL. Never edit old lines.
Read with `tail -n 30 ledger/log.md`, search with `grep -n "E4" ledger/log.md`.

2026-11-03 | francesco/claude | setup | DONE | repo scaffold: AGENTS.md, ledger, uv env (py3.11, upstream pins), CSRv2 submodule @3b2d4d2 | AGENTS.md
2026-11-03 | francesco/claude | setup | NOTE | paper Table 2 & 4 use e5-Mistral-7B; Table 3 is Qwen3-4B; scope set by D-001 | ledger/decisions.md
2026-11-03 | francesco/claude | upstream | BUG | text/data_preparation/get_embeddings_of_each_dataset.py does not parse (IndentationError l.112: docstring/body indent mismatch); we must reimplement it in src/ (E3) | third_party/CSRv2@3b2d4d2
2026-11-03 | francesco/claude | upstream | NOTE | upstream scripts read ./*.json and write ./ckpt ./CSR_results_* relative to CWD and spawn bare `python`: run them via scripts/upstream_rundir.sh under `uv run` | scripts/upstream_rundir.sh
2026-11-03 | francesco/claude | setup | NOTE | released ckpt Y-Research-Group/CSRv2-retrieval = Qwen3-4B + LoRA + SAE(h=10240), k adjustable -> E0/E6b | docs/reproduction/plan.md
