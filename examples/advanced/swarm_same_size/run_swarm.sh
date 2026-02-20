#!/bin/bash
# Adaptive swarm learning with periodic grid search.
# Replaces the old static run_swarm.sh (fixed hyperparameters).
#
# New flow:
#   For each config (5nodes, 10nodes, 20nodes, 40nodes):
#     Every 5 nodes: grid search (swarm, 3 random iters, all lr/bs combos)
#     Then run swarm for all nodes in that group with best hyperparameters
#
# Usage:
#   ./run_swarm.sh                        # both datasets, all configs
#   ./run_swarm.sh --dataset mimic_iii    # single dataset
#   ./run_swarm.sh --config 10nodes       # single config
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

python3 -u "$SCRIPT_DIR/run-swarm-adaptive.py" "$@"
