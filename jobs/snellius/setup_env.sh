#!/usr/bin/env bash
# One-time setup on a Snellius login node:  bash jobs/snellius/setup_env.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
source jobs/snellius/env.sh
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
git submodule update --init --recursive
uv sync --frozen
mkdir -p outputs/slurm  # SLURM needs the --output dir to exist before the job starts
uv run python -c "import torch, sentence_transformers, mteb; print('torch', torch.__version__, '| st', sentence_transformers.__version__, '| mteb', mteb.__version__)"
