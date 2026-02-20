#!/bin/bash
# Adaptive local training with per-configuration grid search.
# Replaces the old static run_local.sh (fixed hyperparameters).
#
# New flow:
#   For each config (5nodes, 10nodes, 20nodes, 40nodes):
#     One grid search per config (local, 3 random iters, all lr/bs combos)
#     Then run local for all num_nodes with best hyperparameters
#
# Usage:
#   ./run_local.sh                        # both datasets, all configs
#   ./run_local.sh --dataset mimic_iii    # single dataset
#   ./run_local.sh --config 10nodes       # single config
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

python3 -u "$SCRIPT_DIR/run-local-adaptive.py" "$@"
