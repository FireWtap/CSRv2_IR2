# Experiment plan

Backbone: `Qwen/Qwen3-Embedding-4B` everywhere. Tasks: the 8 retrieval tasks in `targets.md`.
k grid: {2, 4, 16, 64, 2560} (Table 3); ablation adds {8, 32}. Each experiment id is used in
`ledger/`, `configs/<ID>_*.toml`, `jobs/snellius/<ID>_*.job` and `results/<ID>/`.

| ID  | What                                                                 | Paper ref        | Depends on | Status  |
|-----|----------------------------------------------------------------------|------------------|------------|---------|
| E0  | Released `CSRv2-retrieval` ckpt on NFCorpus+SciFact, all k, fp32+bf16; upstream path == cached path check | Tab.3 (per-task n/a) | env | done (results/E0) |
| E1p | Precision check bf16 vs fp32: dense Qwen3-4B + E0 ckpt on NFCorpus+SciFact -> results/E1p/precision_report.json | D-005 | env | done (results/E1p) |
| E1  | Dense backbone, 8 tasks, prompt protocol both + query, cache embeddings | Tab.3 row 1      | E1p (dtype) | done (results/E1, fp32) |
| E2  | MRL: truncate cached E1 embeddings to k                               | Tab.3 MRL        | E1         | done (results/E2, fp32) |
| E3  | Training embeddings: Qwen3-4B on retrieval train splits (replaces broken upstream `get_embeddings_of_each_dataset.py`) | App. B | env | todo |
| E4  | CSR (no anneal, unsupervised CL) on E3, eval on cached E1            | Tab.3 CSR        | E1, E3     | todo    |
| E5  | CSRv2-linear (anneal + SpSCL) on E3                                   | Tab.3 CSRv2-lin  | E1, E3     | todo    |
| E6  | CSRv2: TopK-LoRA backbone finetune, re-embed, train SAE, eval         | Tab.3 CSRv2      | E3         | todo    |
| E6b | Released `CSRv2-retrieval` ckpt, all 8 tasks x all k                  | Tab.3 CSRv2      | E0         | done (results/E6b, fp32) |
| E7  | Retrieval time, 1M-doc DB, relative to CSRv2@k=2                       | Tab.2 time col   | E2, E5/E6  | todo    |
| E8  | Ablation: CSR, +anneal, +supervise, CSRv2-linear, CSRv2 x k grid      | Tab.4 (ported)   | E4-E6      | optional|
| E9  | Other 5 task types (classif., clust., STS, pair, rerank): dense, MRL, released CSRv2-<type> ckpts, all k (D-008) | Tab.3 other cols | E1, E6b | in progress |

Order of attack: E0 + E1p first (cheapest end-to-end signal that our eval matches theirs), then E1/E2
(no training), then E3 -> E4/E5, then E6, then E7/E8. E6b gives an upper bound for E6 if our
finetune diverges from theirs.

Status lives in `ledger/claims.md` (in progress) and `ledger/log.md` (done); update the
Status column here only when an experiment is finished and its numbers are in `results/`.
