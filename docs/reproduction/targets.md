# Reproduction targets (transcribed from arXiv 2602.05735, `main.tex`)

Read this instead of the PDF. Retrieval metric = mean nDCG@10 over the 8 retrieval tasks below.
"Active dim" = k (number of non-zeros) for CSR-family; truncation length for MRL.

## Retrieval tasks (paper App. A; upstream `text/task_list.json` names)
ArguAna (`arguana`), NFCorpus (`nfcorpus`), CQADupstackGaming (`cqadupstack-gaming`),
CQADupstackUnix (`cqadupstack-unix`), ClimateFEVERHardNegatives (`ClimateFEVER_test_top_250_only_w_correct-v2`),
FiQA2018 (`fiqa`), SCIDOCS (`scidocs`), SciFact (`scifact`). Test splits, English only.

## PRIMARY — Table 3, Qwen3-Embedding-4B, Retrieval column (nDCG@10)
| k    | Backbone | MRL   | CSR   | CSRv2-linear | CSRv2 |
|------|----------|-------|-------|--------------|-------|
| 2560 | 58.37    | 58.31 | 58.21 | 58.74        | 58.23 |
| 64   |          | 44.13 | 51.51 | 53.22        | 55.64 |
| 16   |          | 20.40 | 35.66 | 39.09        | 45.09 |
| 4    |          | 6.59  | 16.27 | 24.82        | 29.71 |
| 2    |          | 5.23  | 9.37  | 18.47        | 23.93 |

Claim under test: CSRv2 > CSRv2-linear > CSR > MRL at equal k, gap largest at k<=4;
"CSRv2 at k=16 rivals MRL at k=64, CSRv2 at k=2 rivals MRL at k=16" (all-task average; check for retrieval).

## PRIMARY (D-008) — Table 3, other task types, Backbone / MRL / CSRv2 rows (E9)
Metrics: Classif. accuracy, Clust. V-measure, STS cosine Spearman, PairClassifi. cosine AP, Rerank. MAP (all x100).
Task lists: `src/csrv2_repro/common.py` TASK_TYPES (10 / 8 / 10 / 2 / 3 tasks). Same numbers are in configs/E9_*.toml.
| k    | Method   | Classif. | Clust. | STS   | PairCl. | Rerank. |
|------|----------|----------|--------|-------|---------|---------|
| 2560 | Backbone | 85.79    | 55.27  | 88.63 | 91.42   | 72.03   |
| 2560 | MRL      | 85.38    | 55.04  | 88.02 | 91.27   | 71.64   |
| 2560 | CSRv2    | 85.58    | 55.91  | 88.47 | 91.39   | 71.98   |
| 64   | MRL      | 83.42    | 53.73  | 86.60 | 88.08   | 69.61   |
| 64   | CSRv2    | 84.28    | 54.57  | 86.32 | 90.90   | 71.64   |
| 16   | MRL      | 75.22    | 47.24  | 79.21 | 73.29   | 60.82   |
| 16   | CSRv2    | 82.03    | 53.86  | 82.63 | 90.42   | 69.89   |
| 4    | MRL      | 48.27    | 32.08  | 53.11 | 30.73   | 40.59   |
| 4    | CSRv2    | 80.20    | 48.27  | 77.94 | 82.28   | 62.98   |
| 2    | MRL      | 26.47    | 24.20  | 30.22 | 18.46   | 32.43   |
| 2    | CSRv2    | 76.22    | 46.02  | 74.88 | 75.24   | 59.52   |

## SECONDARY — Table 2 (e5-Mistral-7B; we port to Qwen3-4B, see ledger D-001)
Retrieval nDCG@10 and retrieval time relative to CSRv2@k=2 (1M-doc database built from all
retrieval + reranking corpora; queries = all retrieval + reranking queries; App. B.1).
| k    | Method       | Ret. time | nDCG@10 |
|------|--------------|-----------|---------|
| 4096 | Backbone     | 306.46x   | 49.35   |
| 4096 | MRL          | 301.86x   | 48.75   |
| 4096 | CSR          | 197.52x   | 49.13   |
| 4096 | CSRv2-linear | 196.04x   | 49.07   |
| 4096 | CSRv2        | 201.42x   | 49.16   |
| 64   | MRL          | 17.30x    | 44.11   |
| 64   | CSR          | 14.92x    | 45.22   |
| 64   | CSRv2-linear | 14.53x    | 47.92   |
| 64   | CSRv2        | 14.17x    | 47.92   |
| 16   | MRL          | 7.77x     | 34.33   |
| 16   | CSR          | 3.53x     | 34.79   |
| 16   | CSRv2-linear | 3.51x     | 39.60   |
| 16   | CSRv2        | 3.51x     | 43.38   |
| 4    | MRL          | 6.29x     | 24.55   |
| 4    | CSR          | 1.62x     | 23.54   |
| 4    | CSRv2-linear | 1.65x     | 34.31   |
| 4    | CSRv2        | 1.63x     | 39.04   |
| 2    | MRL          | 6.20x     | 16.63   |
| 2    | CSR          | 1.01x     | 16.14   |
| 2    | CSRv2-linear | 1.01x     | 31.58   |
| 2    | CSRv2        | 1.00x     | 37.48   |
Port to Qwen3-4B: we compare *relative* speedups and the shape of the trend, not absolute numbers.

## OPTIONAL — Table 4 ablation (e5-Mistral-7B, AVERAGE over all 6 task types)
No retrieval-only numbers are published (only a bar plot, App. Fig. "Retrieval"). Our Qwen3-4B
retrieval-only ablation therefore has no paper reference; we test the *ordering* claim:
CSR < +anneal (small gain) < +supervise < CSRv2-linear (anneal+supervise) < CSRv2 (+finetune), gains grow as k shrinks.
| Method       | anneal | supervise | finetune | k=64  | 32    | 16    | 8     | 4     | 2     |
|--------------|--------|-----------|----------|-------|-------|-------|-------|-------|-------|
| MRL          | -      | -         | yes      | 61.47 | 56.37 | 51.85 | 47.09 | 40.83 | 33.81 |
| CSR          | no     | no        | no       | 66.68 | 64.60 | 61.38 | 58.19 | 52.94 | 44.17 |
| + anneal     | yes    | no        | no       | 67.35 | 65.24 | 61.91 | 58.79 | 54.55 | 45.33 |
| + supervise  | no     | yes       | no       | 67.32 | 65.54 | 62.95 | 60.05 | 56.36 | 49.05 |
| CSRv2-linear | yes    | yes       | no       | 67.58 | 65.83 | 63.73 | 61.53 | 58.62 | 53.25 |
| CSRv2        | yes    | yes       | yes      | 68.08 | 66.70 | 65.22 | 63.76 | 61.01 | 58.34 |

## Hyperparameters for Qwen3-Embedding-4B (paper Tables 6-7, App. B.3)
- CSRv2 SAE: d=2560, h=10240, lr=2e-5, **epoch=128 (suspicious, likely swapped with batch; see D-003)**,
  batch=256, k_aux=1024, beta(aux)=0.1, gamma(SpSCL)=1, K trained = {2,4,16,64,4096}, weight decay 1e-4, Adam.
- Loss: L(k_t) + 1/8 L(4k_t) + beta L_aux + gamma L_SpSCL(k_t). Tied encoder/decoder.
- k-annealing: k_init = 64 if k < 64 else 4k; linear decay reaching target k at 70% of steps.
- Backbone finetune (CSRv2 only, not -linear): LoRA r=8, alpha=16, lr=2e-5, 10 epochs, batch 256, warmup 1000,
  wd 0.1, TopK on backbone output with k in {1,2,...,2048,2560}, uniform weights, InfoNCE (MNRL).
- MRL baseline on Qwen3-4B: native MRL, no finetuning (truncate the 2560-d embedding).
- Inference: encoder -> TopK, no normalization; similarity = cosine (MTEB default).
- Released checkpoint: `Y-Research-Group/CSRv2-retrieval` (Qwen3-4B backbone + LoRA adapter + SAE; k set in
  `3_SparseAutoEncoder/config.json`, default 2). Prompts: upstream `text/dataset_to_prompt.json`.
