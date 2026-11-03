# Source this at the top of every Snellius job (and once interactively before setup).
# Keeps model weights, datasets and caches on scratch instead of the 200GB home quota.
module purge
module load 2024

export SCRATCH_ROOT="${SCRATCH_ROOT:-/scratch-shared/$USER/csrv2}"
export HF_HOME="$SCRATCH_ROOT/hf"
export UV_CACHE_DIR="$SCRATCH_ROOT/uv-cache"
export WANDB_DIR="$SCRATCH_ROOT/wandb"
export WANDB_MODE="${WANDB_MODE:-offline}"
export TOKENIZERS_PARALLELISM=false
mkdir -p "$HF_HOME" "$UV_CACHE_DIR" "$WANDB_DIR"
export PATH="$HOME/.local/bin:$PATH"
