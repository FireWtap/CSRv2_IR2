# Claims — work in progress (edit in place, keep short)

One row per active claim. Add a row BEFORE starting; delete it when done (and append to log.md).
A claim older than 48h without a log entry is stale: ask its owner before taking it over.

| Since (UTC date) | Owner (person/agent) | Exp/area | Paths touched | Branch |
|------------------|----------------------|----------|---------------|--------|
| 2026-11-03 | francesco/claude | E0, E1, E2, E6b | src/csrv2_repro/, configs/E0_*, configs/E1_*, configs/E2_*, configs/E6b_*, jobs/snellius/E0_*, jobs/snellius/E1_*, jobs/snellius/eval8_* | francesco/E0-E1-eval |
| 2026-11-11 | codex | E5 full single linear head | src/csrv2_repro/sae_train.py, configs/E5_linear_full.toml, jobs/snellius/E5_linear_*.job, tests/test_sae_train.py, docs/reproduction/sae_pilot.md, docs/reproduction/targets.md, ledger/, results/E5/linear_full_fp32/ | codex/E3-missing-queries |
