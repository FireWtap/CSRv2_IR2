#!/usr/bin/env bash
# Create an isolated working dir for running upstream CSRv2 text scripts without writing into
# third_party/. Upstream reads ./task_list.json etc. and writes ./ckpt, ./CSR_results_* relative
# to the CWD, so we symlink the JSON files into outputs/runs/<run_id>/ and run from there.
#
# Usage:  RUN=$(scripts/upstream_rundir.sh <run_id>) && cd "$RUN" && \
#         uv run --project "$REPO" python "$UP/all_step_pipeline_mteb_evaluation.py" \
#             --training_script "$UP/CSR_training.py" ...
#   where REPO is the repo root and UP=$REPO/third_party/CSRv2/text
set -euo pipefail
[[ $# -eq 1 ]] || { echo "usage: $0 <run_id>" >&2; exit 1; }
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
UP="$ROOT/third_party/CSRv2/text"
RUN="$ROOT/outputs/runs/$1"
mkdir -p "$RUN"
for f in "$UP"/*.json; do ln -sf "$f" "$RUN/$(basename "$f")"; done
echo "$RUN"
