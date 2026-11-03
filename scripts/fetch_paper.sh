#!/usr/bin/env bash
# Download the CSRv2 paper (arXiv 2602.05735): PDF + LaTeX source into docs/paper/ (gitignored).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/docs/paper"
mkdir -p "$OUT/latex_src"
curl -sSL -o "$OUT/CSRv2_2602.05735.pdf" https://arxiv.org/pdf/2602.05735
curl -sSL https://arxiv.org/e-print/2602.05735 | tar -xz -C "$OUT/latex_src"
echo "Paper in $OUT (tables are in latex_src/main.tex; transcribed targets in docs/reproduction/targets.md)"
