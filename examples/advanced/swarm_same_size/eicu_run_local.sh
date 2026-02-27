#!/bin/bash
# Adaptive local training for eICU dataset with per-configuration grid search.
#
# New flow:
#   For each config (5nodes, 10nodes, 20nodes, 25nodes):
#     One grid search per config (local, 3 random iters, all lr/bs combos)
#     Then run local for all num_nodes with best hyperparameters
#
# Usage:
#   ./eicu_run_local.sh                        # all configs
#   ./eicu_run_local.sh --config 10nodes       # single config
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

python3 -u "$SCRIPT_DIR/eicu_run-local-adaptive.py" "$@"
