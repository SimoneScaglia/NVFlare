#!/bin/bash
# Adaptive central training with periodic grid search.
# Replaces the old static run_central.sh (fixed hyperparameters).
#
# New flow:
#   For each config (5nodes, 10nodes, 20nodes, 40nodes):
#     Every 5 nodes: grid search (central, 3 random iters, all lr/bs combos)
#     Then run central for all nodes in that group with best hyperparameters
#
# Usage:
#   ./run_central.sh                        # both datasets, all configs
#   ./run_central.sh --dataset mimic_iii    # single dataset
#   ./run_central.sh --config 10nodes       # single config
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

python3 -u "$SCRIPT_DIR/run-central-adaptive.py" "$@"
