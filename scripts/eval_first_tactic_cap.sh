#!/usr/bin/env bash
# Sweep --first-token-occurrences-cap over {none, 1, 2, 3, 6} for one model.
#
# Usage:
#   scripts/eval_first_tactic_cap.sh <model-path> --lean-servers <addr> [extra prover_eval args...]

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <model-path> [extra prover_eval args...]" >&2
    exit 1
fi

model_path="$1"
shift

caps=(none 2 1 3 6)

for cap in "${caps[@]}"; do
    suffix="_first-tok-cap=${cap}"
    echo "================================================================"
    echo "Running with --first-token-occurrences-cap=${cap}  (suffix=${suffix})"
    echo "================================================================"
    OMP_NUM_THREADS=1 torchrun --standalone --nproc_per_node=8 \
        -m scripts.prover_eval \
        --model-path "$model_path" \
        --first-token-occurrences-cap "$cap" \
        --output-suffix "$suffix" \
        "$@"
done
