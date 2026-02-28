#!/bin/bash
# Adaptive swarm learning for eICU dataset with periodic grid search.
#
# New flow:
#   For each config (5nodes, 10nodes, 20nodes, 25nodes):
#     Every 5 nodes: grid search (swarm, 3 random iters, all lr/bs combos)
#     Then run swarm for all num_nodes in that group with best hyperparameters
#
# Usage:
#   ./eicu_run_swarm.sh                        # all configs
#   ./eicu_run_swarm.sh --config 10nodes       # single config
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

source "$SCRIPT_DIR/../swarm_learning_tf/swarm_env/bin/activate"
python3 -u "$SCRIPT_DIR/eicu_run-swarm-adaptive.py" "$@"
