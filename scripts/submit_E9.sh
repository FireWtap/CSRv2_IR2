#!/bin/bash
# Submit all E9 jobs (ledger D-008): per non-retrieval task type, one dense pass and one released-CSRv2 pass
# on gpu_a100, plus an MRL scoring job on gpu_mig that starts after its dense pass succeeds.
#   bash scripts/submit_E9.sh [type ...]     (default: all five)
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
mkdir -p outputs/slurm
TYPES=("${@:-classification clustering sts pair_classification reranking}")
for t in ${TYPES[@]}; do
  d=$(sbatch --parsable --job-name="e9_dense_$t" --export=ALL,PART=dense,TASK_TYPE="$t" jobs/snellius/E9_pass.job)
  c=$(sbatch --parsable --job-name="e9_csrv2_$t" --export=ALL,PART=csrv2,TASK_TYPE="$t" jobs/snellius/E9_pass.job)
  m=$(sbatch --parsable --job-name="e9_mrl_$t" --dependency=afterok:"$d" --export=ALL,TASK_TYPE="$t" jobs/snellius/E9_mrl.job)
  echo "$t dense=$d csrv2=$c mrl=$m"
done
