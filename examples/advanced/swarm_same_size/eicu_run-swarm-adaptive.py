#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Adaptive swarm learning for eICU dataset with periodic grid search.

Key differences from MIMIC version:
  - Data is per-hospital: {hospitalid}_train.csv and {hospitalid}_test.csv
  - Each iteration randomly selects num_nodes hospitals from the available pool
  - Each site's model is evaluated on that site's hospital-specific test data
  - Hospital selection is deterministic per iteration (for reproducibility),
    and smaller num_nodes selections are always subsets of larger ones

Workflow:
  For each configuration (5nodes, 10nodes, 20nodes, 25nodes):
    For each group of 5 num_nodes (boundary at 5, 10, 15, ...):
      1. Grid search: run swarm at the boundary num_nodes
         with 3 random iterations and all lr/bs combinations
      2. Pick best (lr, bs) based on average AUC across sites/iterations
      3. Save grid search results (heatmap-compatible CSV)
      4. Run swarm for all num_nodes in the group (all 10 iterations)
         using the best hyperparameters
      5. Save swarm results to results/eicu/swarm_results.csv

    Grid search hyperparameters:
      batch_sizes = [16, 32, 64, 128, 256, 512]
      lrs = [0.0001, 0.001, 0.01, 0.1]

    Group assignment example (for 20nodes config):
      Grid search at n=5  -> best params used for num_nodes = 2,3,4,5
      Grid search at n=10 -> best params used for num_nodes = 6,7,8,9,10
      Grid search at n=15 -> best params used for num_nodes = 11,12,13,14,15
      Grid search at n=20 -> best params used for num_nodes = 16,17,18,19,20

Usage:
    python eicu_run-swarm-adaptive.py                     # all configs
    python eicu_run-swarm-adaptive.py --config 10nodes    # single config
"""

import argparse
import gc
import os
import random
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

# ── Constants ──────────────────────────────────────────────────────────────────
SEED = 42
EPOCHS = 25
NUM_ITERATIONS = 10           # total iterations per configuration (0-9)
GRID_SEARCH_ITERS = 3        # random iterations used in grid search
GROUP_SIZE = 5                # re-tune hyperparameters every N nodes

BATCH_SIZES = [16, 32, 64, 128, 256, 512]
LRS = [0.0001, 0.001, 0.01, 0.1]

CONFIGS = {
    "5nodes":  5,
    "10nodes": 10,
    "20nodes": 20,
    "25nodes": 25,
}

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EICU_DATA_DIR = os.path.join(SCRIPT_DIR, "datasets", "eicu", "data_fixed_rows")
SWARM_TF_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "swarm_learning_tf"))
CONFIG_FILE = os.path.normpath(os.path.join(
    SCRIPT_DIR, "..", "..", "..", "job_templates",
    "swarm_cse_tf_model_learner", "config_fed_client.conf"
))

# Add mimic networks to path for FCN model
sys.path.insert(0, os.path.join(SWARM_TF_DIR, "..", "mimic", "networks"))
from mimic_nets import FCN  # noqa: E402


# ── Reproducibility ───────────────────────────────────────────────────────────
def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['PYTHONHASHSEED'] = str(seed)
    tf.config.experimental.enable_op_determinism()


set_seed()


# ── Hospital selection ─────────────────────────────────────────────────────────
def get_available_hospitals(config_name):
    """Get sorted list of available hospital IDs for a configuration."""
    data_dir = os.path.join(EICU_DATA_DIR, config_name)
    hospitals = []
    for f in os.listdir(data_dir):
        if f.endswith("_train.csv"):
            hospital_id = int(f.replace("_train.csv", ""))
            hospitals.append(hospital_id)
    return sorted(hospitals)


def select_hospitals(available_hospitals, num_nodes, iteration):
    """
    Deterministically select num_nodes hospitals for a given iteration.

    The selection ensures that smaller num_nodes is always a subset
    of larger num_nodes for the same iteration.
    """
    rng = random.Random(SEED + iteration)
    shuffled = available_hospitals.copy()
    rng.shuffle(shuffled)
    return sorted(shuffled[:num_nodes])


# ── Shell command execution ───────────────────────────────────────────────────
def run_command(command, cwd=None):
    """Execute a shell command with swarm_env activated."""
    if cwd is None:
        cwd = SWARM_TF_DIR
    activate_env = f"source {SWARM_TF_DIR}/swarm_env/bin/activate && "
    full_command = activate_env + command
    result = subprocess.run(
        full_command, shell=True, capture_output=True, text=True,
        cwd=cwd, executable='/bin/bash'
    )
    if result.returncode != 0:
        print(f"  [CMD ERROR] {command}")
        if result.stdout:
            print(f"  STDOUT (last 500 chars): {result.stdout[-500:]}")
        if result.stderr:
            print(f"  STDERR (last 500 chars): {result.stderr[-500:]}")
    return result


# ── NVFlare config editing ────────────────────────────────────────────────────
def edit_config_file(min_responses_required, learning_rate, batch_size):
    """Edit the NVFlare config_fed_client.conf with given hyperparameters."""
    with open(CONFIG_FILE, 'r') as f:
        content = f.read()

    content = re.sub(
        r'min_responses_required\s*=\s*\d+',
        f'min_responses_required = {min_responses_required}',
        content
    )

    learner_pattern = r'(id = "mimic-learner"[^{]*args \{)([^}]+)(\})'

    def update_learner_args(match):
        args_section = match.group(2)
        args_section = re.sub(r'lr\s*=\s*[\d.eE+-]+', f'lr = {learning_rate}', args_section)
        args_section = re.sub(r'batch_size\s*=\s*\d+', f'batch_size = {batch_size}', args_section)
        return match.group(1) + args_section + match.group(3)

    content = re.sub(learner_pattern, update_learner_args, content, flags=re.DOTALL)

    with open(CONFIG_FILE, 'w') as f:
        f.write(content)


# ── Data file management ─────────────────────────────────────────────────────
def copy_data_files(config_name, selected_hospitals):
    """
    Copy selected hospital train CSV files to /tmp/mimic_data/ as site-{i}.csv.

    Returns a mapping {site_id: hospital_id} for evaluation.
    """
    dest = "/tmp/mimic_data"
    os.makedirs(dest, exist_ok=True)

    # Clean old files
    for f in os.listdir(dest):
        os.remove(os.path.join(dest, f))

    data_dir = os.path.join(EICU_DATA_DIR, config_name)
    site_to_hospital = {}

    for i, hid in enumerate(selected_hospitals, start=1):
        src = os.path.join(data_dir, f"{hid}_train.csv")
        dst = os.path.join(dest, f"site-{i}.csv")
        if os.path.isfile(src):
            shutil.copy(src, dst)
            site_to_hospital[i] = hid
        else:
            print(f"  WARNING: {src} does not exist!")

    return site_to_hospital


# ── Metrics helpers ───────────────────────────────────────────────────────────
def get_metrics_fns():
    return [
        tf.keras.metrics.AUC(name='auc', curve='ROC', num_thresholds=1000),
        tf.keras.metrics.AUC(name='auprc', curve='PR', num_thresholds=1000),
        tf.keras.metrics.BinaryAccuracy(name='accuracy'),
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall'),
    ]


# ── Core swarm experiment ─────────────────────────────────────────────────────
def run_swarm_experiment(config_name, num_nodes, iteration, lr, bs, available_hospitals):
    """
    Run one swarm learning experiment via NVFlare simulator.

    Each site's model is evaluated on that site's hospital-specific test data.
    Returns a list of (site_id, hospital_id, metrics_dict) tuples.
    """
    selected = select_hospitals(available_hospitals, num_nodes, iteration)
    data_dir = os.path.join(EICU_DATA_DIR, config_name)

    # 1. Copy data files
    site_to_hospital = copy_data_files(config_name, selected)

    # 2. Edit NVFlare config
    edit_config_file(num_nodes, lr, bs)

    # 3. Sync code
    run_command("rsync -av --exclude='dataset' ../mimic ./code/", cwd=SWARM_TF_DIR)

    # 4. Create NVFlare job
    job_name = f"eicu_swarm_{num_nodes}"
    run_command(
        f"nvflare job create -j ./jobs/{job_name} "
        f"-w swarm_cse_tf_model_learner -sd ./code -force",
        cwd=SWARM_TF_DIR
    )

    # 5. Run NVFlare simulator
    run_command(
        f"nvflare simulator ./jobs/{job_name} "
        f"-w /tmp/nvflare/{job_name} -n {num_nodes} -t {num_nodes}",
        cwd=SWARM_TF_DIR
    )

    # 6. Evaluate each site's global model on its hospital's test data
    site_results = []
    models_path = f"/tmp/nvflare/{job_name}"

    for site_id, hid in site_to_hospital.items():
        # Load this hospital's test data
        test_file = os.path.join(data_dir, f"{hid}_test.csv")
        if not os.path.exists(test_file):
            print(f"  Test file not found: {test_file}")
            continue

        test_df = pd.read_csv(test_file)
        X_test = test_df.iloc[:, :-1].values.astype(np.float32)
        y_test = test_df.iloc[:, -1].values.astype(np.float32)
        input_dim = X_test.shape[1]

        model_path = os.path.join(
            models_path, f'site-{site_id}', 'simulate_job',
            f'app_site-{site_id}', f'site-{site_id}.weights.h5'
        )
        if not os.path.exists(model_path):
            print(f"  Model not found: {model_path}")
            continue

        try:
            tf.keras.backend.clear_session()
            model = FCN(input_dim=input_dim)
            model.build((None, input_dim))
            model.load_weights(model_path)
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                loss='binary_crossentropy',
                metrics=get_metrics_fns()
            )

            metrics = model.evaluate(X_test, y_test, verbose=0, return_dict=True)
            site_results.append((site_id, hid, metrics))
        except Exception as e:
            print(f"  Error evaluating site-{site_id} (hospital {hid}): {e}")
        finally:
            gc.collect()

    # 7. Clean up simulator output
    if os.path.exists(models_path):
        shutil.rmtree(models_path)

    return site_results, selected


# ── Grid search ───────────────────────────────────────────────────────────────
def grid_search(config_name, num_nodes, available_hospitals):
    """
    Run swarm grid search at a boundary num_nodes.

    Tries all (batch_size, learning_rate) combinations with 3 random iterations.
    Returns (best_lr, best_bs) and saves results CSV for heatmap.
    """
    gs_dir = os.path.join(SCRIPT_DIR, "datasets", "eicu", "data_fixed_rows_results", config_name)
    os.makedirs(gs_dir, exist_ok=True)
    gs_file = os.path.join(gs_dir, f"grid_search_n{num_nodes}.csv")

    if os.path.exists(gs_file):
        print(f"\n  Grid search already completed: {gs_file}")
        gs_df = pd.read_csv(gs_file)
        if len(gs_df) > 0:
            best_lr, best_bs = _find_best_params(gs_df)
            print(f"  Loaded best params: LR={best_lr} BS={best_bs}")
            return best_lr, best_bs
        print(f"  File empty, re-running grid search...")

    print(f"\n{'='*60}")
    print(f"SWARM GRID SEARCH (eICU): config={config_name} num_nodes={num_nodes}")
    print(f"Hyperparameters: BS={BATCH_SIZES} LR={LRS}")
    print(f"{'='*60}")

    # Pick 3 random iterations (deterministic per boundary for reproducibility)
    rng = random.Random(SEED + num_nodes + hash(config_name))
    iterations = sorted(rng.sample(range(NUM_ITERATIONS), GRID_SEARCH_ITERS))
    print(f"Selected iterations for grid search: {iterations}")

    gs_results = []
    total = len(BATCH_SIZES) * len(LRS) * len(iterations)
    done = 0

    for bs in BATCH_SIZES:
        for lr in LRS:
            combo_aucs = []
            for iteration in iterations:
                done += 1
                selected = select_hospitals(available_hospitals, num_nodes, iteration)
                print(f"\n  [{done}/{total}] BS={bs} LR={lr} Iter={iteration} "
                      f"(n={num_nodes}, hospitals={selected})")

                site_results, _ = run_swarm_experiment(
                    config_name, num_nodes, iteration, lr, bs, available_hospitals
                )

                for site_id, hid, m in site_results:
                    gs_results.append({
                        'num_nodes': num_nodes,
                        'iteration': iteration,
                        'site_id': site_id,
                        'hospital_id': hid,
                        'batch_size': bs,
                        'learning_rate': lr,
                        'loss': m['loss'],
                        'auc': m['auc'],
                        'auprc': m['auprc'],
                        'accuracy': m['accuracy'],
                        'precision': m['precision'],
                        'recall': m['recall'],
                    })
                    combo_aucs.append(m['auc'])

            avg_auc = np.mean(combo_aucs) if combo_aucs else 0
            print(f"  -> Avg AUC for BS={bs} LR={lr}: {avg_auc:.4f}")

    # Save grid search results
    gs_df = pd.DataFrame(gs_results)
    gs_df.to_csv(gs_file, index=False)
    print(f"\nGrid search results saved to {gs_file}")

    # Find best hyperparameters
    best_lr, best_bs = _find_best_params(gs_df)
    print(f"\n*** BEST HYPERPARAMETERS: LR={best_lr} BS={best_bs} ***")

    return best_lr, best_bs


def _find_best_params(gs_df):
    """Find best (lr, bs) from grid search results based on mean AUC."""
    grouped = gs_df.groupby(['batch_size', 'learning_rate'])['auc'].mean()
    best_idx = grouped.idxmax()
    best_bs, best_lr = best_idx
    best_auc = grouped[best_idx]
    print(f"  Best combo: LR={best_lr} BS={best_bs} (mean AUC={best_auc:.4f})")
    return best_lr, int(best_bs)


# ── Swarm runs for a group ────────────────────────────────────────────────────
def run_swarm_group(config_name, num_nodes_range, lr, bs, available_hospitals):
    """
    Run swarm experiments for a group of num_nodes using given hyperparameters.
    All 10 iterations are executed. Results are appended to swarm_results.csv.
    """
    print(f"\n{'='*60}")
    print(f"SWARM RUNS (eICU): config={config_name} "
          f"nodes={list(num_nodes_range)} LR={lr} BS={bs}")
    print(f"{'='*60}")

    results_dir = os.path.join(SCRIPT_DIR, "results", "eicu")
    os.makedirs(results_dir, exist_ok=True)
    results_file = os.path.join(results_dir, "swarm_results.csv")

    # Load or create results dataframe
    if os.path.exists(results_file):
        results_df = pd.read_csv(results_file)
    else:
        results_df = pd.DataFrame(columns=[
            'datetime', 'configuration', 'num_nodes', 'site_id', 'hospital_id',
            'iteration', 'lr', 'batch_size',
            'loss', 'auc', 'auprc', 'accuracy', 'precision', 'recall'
        ])

    for num_nodes in num_nodes_range:
        for iteration in range(NUM_ITERATIONS):
            # Check if already computed (resumability)
            existing = results_df[
                (results_df['configuration'] == config_name) &
                (results_df['num_nodes'] == num_nodes) &
                (results_df['iteration'] == iteration)
            ]
            if len(existing) > 0:
                print(f"  [SKIP] Config={config_name} Nodes={num_nodes} "
                      f"Iter={iteration} (already in results)")
                continue

            selected = select_hospitals(available_hospitals, num_nodes, iteration)
            print(f"\n--- Config={config_name} Nodes={num_nodes} "
                  f"Iter={iteration} LR={lr} BS={bs} Hospitals={selected} ---")

            site_results, _ = run_swarm_experiment(
                config_name, num_nodes, iteration, lr, bs, available_hospitals
            )

            for site_id, hid, m in site_results:
                new_row = {
                    'datetime': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'configuration': config_name,
                    'num_nodes': num_nodes,
                    'site_id': site_id,
                    'hospital_id': hid,
                    'iteration': iteration,
                    'lr': lr,
                    'batch_size': bs,
                    'loss': m['loss'],
                    'auc': m['auc'],
                    'auprc': m['auprc'],
                    'accuracy': m['accuracy'],
                    'precision': m['precision'],
                    'recall': m['recall'],
                }
                results_df = pd.concat(
                    [results_df, pd.DataFrame([new_row])], ignore_index=True
                )
                print(f"  [Site-{site_id} / Hospital-{hid}] AUC={m['auc']:.4f} "
                      f"AUPRC={m['auprc']:.4f}")

            # Save after each num_nodes/iteration for safety
            results_df.to_csv(results_file, index=False)

    print(f"\nSwarm results saved to {results_file}")


# ── Main orchestrator ─────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Adaptive swarm learning for eICU with periodic grid search"
    )
    parser.add_argument(
        '--config',
        choices=list(CONFIGS.keys()) + ['all'],
        default='all',
        help="Configuration to process (default: all)"
    )
    args = parser.parse_args()

    configs = CONFIGS if args.config == 'all' else {args.config: CONFIGS[args.config]}

    print(f"\n{'#'*60}")
    print(f"ADAPTIVE SWARM LEARNING (eICU) - Started {datetime.now()}")
    print(f"Configurations: {list(configs.keys())}")
    print(f"Grid search: {len(BATCH_SIZES)} batch_sizes x {len(LRS)} lrs "
          f"x {GRID_SEARCH_ITERS} iterations")
    print(f"Group size: {GROUP_SIZE} (re-tune every {GROUP_SIZE} nodes)")
    print(f"{'#'*60}")

    for config_name, max_nodes in configs.items():
        print(f"\n{'='*60}")
        print(f"CONFIGURATION: {config_name} (max_nodes={max_nodes})")
        print(f"{'='*60}")

        # Verify config directory exists
        config_dir = os.path.join(EICU_DATA_DIR, config_name)
        if not os.path.isdir(config_dir):
            print(f"ERROR: Config directory not found: {config_dir}")
            continue

        available_hospitals = get_available_hospitals(config_name)
        print(f"Available hospitals: {len(available_hospitals)} -> {available_hospitals}")

        if len(available_hospitals) < max_nodes:
            print(f"ERROR: Not enough hospitals ({len(available_hospitals)}) "
                  f"for config {config_name} (need {max_nodes})")
            continue

        # Process groups of GROUP_SIZE
        for group_end in range(GROUP_SIZE, max_nodes + 1, GROUP_SIZE):
            group_start = group_end - GROUP_SIZE + 1
            # First group starts at 2 (no point running swarm with 1 node)
            if group_start < 2:
                group_start = 2

            print(f"\n--- Group: num_nodes = [{group_start}..{group_end}] ---")

            # Phase 1: Grid search at boundary
            best_lr, best_bs = grid_search(config_name, group_end, available_hospitals)

            # Phase 2: Run swarm for all nodes in this group
            nodes_range = range(group_start, group_end + 1)
            run_swarm_group(config_name, nodes_range, best_lr, best_bs, available_hospitals)

    print(f"\n{'#'*60}")
    print(f"ALL EXPERIMENTS COMPLETED - {datetime.now()}")
    print(f"{'#'*60}")


if __name__ == "__main__":
    main()
