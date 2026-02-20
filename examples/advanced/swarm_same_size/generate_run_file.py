#!/usr/bin/env python3
"""
Generate shell scripts to run local and central experiments
for same-size configurations.

Configurations: 5nodes, 10nodes, 20nodes, 40nodes
Epochs: 25 (fixed)
Best hyperparameters per configuration:
    5nodes:  lr = 0.0025,  bs = 128
    10nodes: lr = 0.00025, bs = 8
    20nodes: lr = 0.00250, bs = 128
    40nodes: lr = 0.01,    bs = 512

For each configuration, for each iteration (0-9), runs from 2 nodes up to
the maximum number of nodes in that configuration.

Note: Swarm experiments are handled by run-swarm-adaptive.py (with grid search).

Output:
    run_local.sh   - calls run-local.py
    run_central.sh - calls run-central.py
"""

import os

# ── Configuration ──────────────────────────────────────────────────────
CONFIGS = {
    "5nodes":  {"max_nodes": 5,  "lr": 0.0025,   "bs": 128},
    "10nodes": {"max_nodes": 10, "lr": 0.00025,  "bs": 8},
    "20nodes": {"max_nodes": 20, "lr": 0.00250,  "bs": 128},
    "40nodes": {"max_nodes": 40, "lr": 0.01,     "bs": 512},
}

NUM_ITERATIONS = 10  # iterations 0..9

script_dir = os.path.dirname(os.path.abspath(__file__))

# ── run_local.sh ───────────────────────────────────────────────────────
local_path = os.path.join(script_dir, "run_local.sh")
with open(local_path, "w") as f:
    f.write("#!/bin/bash\n")
    f.write("# Auto-generated: local training experiments\n")
    f.write("set -e\n\n")
    f.write('SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"\n\n')

    for config_name, params in CONFIGS.items():
        max_nodes = params["max_nodes"]
        lr = params["lr"]
        bs = params["bs"]
        f.write(f"# ── {config_name} ──\n")
        for num_nodes in range(2, max_nodes + 1):
            for iteration in range(NUM_ITERATIONS):
                f.write(
                    f'python3 -u "$SCRIPT_DIR/run-local.py" '
                    f'{num_nodes} {config_name} {iteration} {lr} {bs}\n'
                )
        f.write("\n")

print(f"Generated: {local_path}")

# ── run_central.sh ─────────────────────────────────────────────────────
central_path = os.path.join(script_dir, "run_central.sh")
with open(central_path, "w") as f:
    f.write("#!/bin/bash\n")
    f.write("# Auto-generated: central training experiments\n")
    f.write("set -e\n\n")
    f.write('SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"\n\n')

    for config_name, params in CONFIGS.items():
        max_nodes = params["max_nodes"]
        lr = params["lr"]
        bs = params["bs"]
        f.write(f"# ── {config_name} ──\n")
        for num_nodes in range(2, max_nodes + 1):
            for iteration in range(NUM_ITERATIONS):
                f.write(
                    f'python3 -u "$SCRIPT_DIR/run-central.py" '
                    f'{num_nodes} {config_name} {iteration} {lr} {bs}\n'
                )
        f.write("\n")

print(f"Generated: {central_path}")

# Make scripts executable
for path in [local_path, central_path]:
    os.chmod(path, 0o755)

print("\nDone. Generated scripts:")
print(f"  {local_path}")
print(f"  {central_path}")
print("\nNote: Swarm experiments are handled by run-swarm-adaptive.py")
