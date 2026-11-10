# How upstream trains the CSRv2 retrieval checkpoint (code audit, 2026-11-03)

Sources: `third_party/CSRv2@3b2d4d2` (text/), released `Y-Research-Group/CSRv2-retrieval@af16198`
(model card, `adapter_config.json`, `3_SparseAutoEncoder/config.json`, `trainer_state.json`).
`training_args.bin` (pickle) was not opened, so effective batch / GPU count are unknown.

## Pipeline
1. **Finetune data** (`data_preparation/get_dataset_for_finetuning.py`): (query, positive doc) pairs from
   the 8 MTEB retrieval datasets themselves, <=20000 pairs per dataset (README). Uses the `train` split if
   the dataset has one, otherwise an 80/20 `train_test_split(seed=42)` **of the test qrels rows**.
2. **Backbone finetune** (`topk_lora_finetuning.py`): LoRA r=8, alpha=16, dropout 0.1 on q/k/v/o/gate/up/down,
   MNRL loss, TopK applied to the backbone output over a k list, uniform weights.
3. **SAE training data** (`data_preparation/get_embeddings_of_each_dataset.py` -> `get_embeddings_for_training.py`,
   `combine_retrieval_embedding`): embeds full corpus + all queries of each dataset; training rows = train-pair
   queries and docs (label = query group) + **every unpaired query and corpus doc** (singleton labels).
   Datasets without a train split: 80% of **test queries** (split by query, seed 42) become training pairs.
   `dev` splits are never merged (code looks for `val.npz`).
4. **SAE training** (`CSR_training.py`, via `all_step_pipeline_mteb_evaluation.py`):
   loss = MSE(k) + 1/8 MSE(4k) + auxk_coef * aux + cl_coef * label-based contrastive (`--use_label_CL`).
   Single `--topk` target per run; annealing from `--initial_topk` linearly to `--topk` at 70% of steps.
5. **Eval**: mteb on the **full** test split of the same 8 datasets.

## Findings
- **Test contamination (5 of 8 tasks).** ArguAna, CQADupstack-Gaming, CQADupstack-Unix,
  ClimateFEVERHardNegatives and SCIDOCS have only a test split on HF, so both the LoRA finetune (80% of test
  qrels rows) and the SAE label-CL (80% of test queries) train on test relevance labels, then are evaluated on
  the full test set. NFCorpus, FiQA2018, SciFact use their train splits. Applies to the CSRv2 and
  CSRv2-linear rows of Table 3, not to the dense/MRL baselines. Corpus docs of all tasks are also in SAE
  training (unsupervised/transductive).
  *Verification:* certain for the released code: the HF repos of those 5 tasks only have test qrels, the
  `else` branch takes 80% of them, the README/training.md finetune command reads exactly this dataset
  (`MTEB_single_task_type_finetuning_dataset/retrieval`), and `eval_all()` scores the full mteb `test` split.
  For the released checkpoint itself it is strong evidence, not proof: re-counting the recipe
  (scratchpad script, current HF data) gives 61806 finetune pairs (35085 train-split + 26721 test-derived);
  `trainer_state.json` implies <=69156 pairs (17289 steps x bs 4, 1 GPU, no accumulation). Train splits only
  (35085 -> ~8.8k steps) cannot explain 17289 steps under the documented 20k cap; with test-derived pairs it
  is close but not exact (gap ~7k, unknown cause: other cap, dataset revisions, or code not in the repo;
  note the repo's `get_embeddings_of_each_dataset.py` does not even parse, so it is not what they ran).
  The SAE-side leakage can only be checked in code, not from the checkpoint.
- **Correction (2026-11-10): multi-TopK is implemented.** The paper explicitly calls
  `L(k) + L(4k)/8` its multi-TopK loss (Eqs. 4, 8); the released trainer implements that term.
  This supports changing inference k with a shared SAE and is not evidence of missing multi-k code.
  Which training target(s) produced every table entry/released head remains undocumented here.
  See [the SAE audit](sae_audit.md) for verified reuse and actual implementation discrepancies.
- **Backbone finetune differs from the paper (Table 7: 10 epochs, lr 2e-5, warmup 1000).** `trainer_state.json`:
  `num_train_epochs=1`, stopped at step 16000 of 17289 (epoch 0.93), per-device batch 4, peak lr 1e-5
  reached at step ~1800 (linear warmup). Trained with sentence-transformers 5.2.0, peft 0.18.1 (we pin 5.0.0 / 0.15.2).
