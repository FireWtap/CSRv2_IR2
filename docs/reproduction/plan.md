# Experiment plan

Backbone: `Qwen/Qwen3-Embedding-4B` everywhere. Tasks: the 8 retrieval tasks in `targets.md`.
k grid: {2, 4, 16, 64, 2560} (Table 3); ablation adds {8, 32}. Each experiment id is used in
`ledger/`, `configs/<ID>_*.yaml`, `jobs/snellius/<ID>_*.job` and `results/<ID>/`.

| ID  | What                                                                 | Paper ref        | Depends on | Status  |
|-----|----------------------------------------------------------------------|------------------|------------|---------|
| E0  | Eval pipeline sanity: released `CSRv2-retrieval` ckpt on NFCorpus, k=2 | Tab.3 (per-task n/a) | env     | todo    |
| E1  | Dense backbone, 8 tasks, cache query/corpus embeddings to `outputs/` | Tab.3 row 1      | env        | todo    |
| E2  | MRL: truncate cached E1 embeddings to k                               | Tab.3 MRL        | E1         | todo    |
| E3  | Training embeddings: Qwen3-4B on retrieval train splits (replaces broken upstream `get_embeddings_of_each_dataset.py`) | App. B | env | todo |
| E4  | CSR (no anneal, unsupervised CL) on E3, eval on cached E1            | Tab.3 CSR        | E1, E3     | todo    |
| E5  | CSRv2-linear (anneal + SpSCL) on E3                                   | Tab.3 CSRv2-lin  | E1, E3     | todo    |
| E6  | CSRv2: TopK-LoRA backbone finetune, re-embed, train SAE, eval         | Tab.3 CSRv2      | E3         | todo    |
| E6b | Released `CSRv2-retrieval` ckpt, all 8 tasks x all k                  | Tab.3 CSRv2      | E0         | todo    |
| E7  | Retrieval time, 1M-doc DB, relative to CSRv2@k=2                       | Tab.2 time col   | E2, E5/E6  | todo    |
| E8  | Ablation: CSR, +anneal, +supervise, CSRv2-linear, CSRv2 x k grid      | Tab.4 (ported)   | E4-E6      | optional|

Order of attack: E0 first (cheapest end-to-end signal that our eval matches theirs), then E1/E2
(no training), then E3 -> E4/E5, then E6, then E7/E8. E6b gives an upper bound for E6 if our
finetune diverges from theirs.

Status lives in `ledger/claims.md` (in progress) and `ledger/log.md` (done); update the
Status column here only when an experiment is finished and its numbers are in `results/`.
