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
Adaptive central training for eICU dataset with periodic grid search.

Key differences from MIMIC version:
  - Data is per-hospital: {hospitalid}_train.csv and {hospitalid}_test.csv
  - Each iteration randomly selects num_nodes hospitals from the available pool
  - Central trains on merged train files, evaluates on merged test files
  - Hospital selection is deterministic per iteration (for reproducibility),
    and smaller num_nodes selections are always subsets of larger ones

Workflow:
  For each configuration (5nodes, 10nodes, 20nodes, 25nodes):
    For each group of 5 num_nodes (boundary at 5, 10, 15, ...):
      1. Grid search at the boundary num_nodes:
         merge data from boundary_nodes hospitals, train centralized model,
         evaluate on merged test with 3 random iterations x all lr/bs combos
      2. Pick best (lr, bs) based on mean AUC
      3. Save grid search results
      4. Run central training for all num_nodes in the group (all 10 iterations)
         using the best hyperparameters
      5. Save results to results/eicu/central_results.csv

    Grid search hyperparameters:
      batch_sizes = [16, 32, 64, 128, 256, 512]
      lrs = [0.0001, 0.001, 0.01, 0.1]

Usage:
    python eicu_run-central-adaptive.py                # all configs
    python eicu_run-central-adaptive.py --config 10nodes
"""

import argparse
import gc
import os
import random
from datetime import datetime

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
    of larger num_nodes for the same iteration (we shuffle once per
    iteration and take the first num_nodes).
    """
    rng = random.Random(SEED + iteration)
    shuffled = available_hospitals.copy()
    rng.shuffle(shuffled)
    return sorted(shuffled[:num_nodes])


# ── Model ─────────────────────────────────────────────────────────────────────
def get_net(input_dim):
    """Build the FCN model."""
    kernel_init = tf.keras.initializers.GlorotUniform(seed=SEED)
    bias_init = tf.keras.initializers.Zeros()
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(16, activation="relu", input_shape=(input_dim,), kernel_initializer=kernel_init, bias_initializer=bias_init),
        tf.keras.layers.Dense(16, activation="relu", kernel_initializer=kernel_init, bias_initializer=bias_init),
        tf.keras.layers.Dense(16, activation="relu", kernel_initializer=kernel_init, bias_initializer=bias_init),
        tf.keras.layers.Dense(1, activation="sigmoid", kernel_initializer=kernel_init, bias_initializer=bias_init),
    ])
    return model


def get_metrics():
    """Build fresh metric objects for model compilation."""
    return [
        tf.keras.metrics.AUC(name='auc', curve='ROC', num_thresholds=1000),
        tf.keras.metrics.AUC(name='auprc', curve='PR', num_thresholds=1000),
        tf.keras.metrics.BinaryAccuracy(name='accuracy'),
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall'),
    ]


# ── Core central training ────────────────────────────────────────────────────
def run_central_experiment(config_name, num_nodes, iteration, lr, bs, available_hospitals):
    """
    Run one central training experiment for eICU.

    Selects num_nodes hospitals for this iteration, merges their train data,
    trains one model, evaluates on merged test data.
    Returns metrics dict or None on failure.
    """
    data_dir = os.path.join(EICU_DATA_DIR, config_name)
    selected = select_hospitals(available_hospitals, num_nodes, iteration)

    # Merge training data
    train_dfs = []
    test_dfs = []
    for hid in selected:
        train_file = os.path.join(data_dir, f"{hid}_train.csv")
        test_file = os.path.join(data_dir, f"{hid}_test.csv")
        if not os.path.exists(train_file) or not os.path.exists(test_file):
            print(f"  WARNING: Files for hospital {hid} not found, skipping")
            continue
        train_dfs.append(pd.read_csv(train_file))
        test_dfs.append(pd.read_csv(test_file))

    if len(train_dfs) == 0:
        print(f"  ERROR: No training data found for config={config_name} iter={iteration}")
        return None, selected

    merged_train = pd.concat(train_dfs, ignore_index=True)
    merged_test = pd.concat(test_dfs, ignore_index=True)

    X_train = merged_train.iloc[:, :-1].values.astype(np.float32)
    y_train = merged_train.iloc[:, -1].values.astype(np.float32)
    X_test = merged_test.iloc[:, :-1].values.astype(np.float32)
    y_test = merged_test.iloc[:, -1].values.astype(np.float32)
    input_dim = X_train.shape[1]

    try:
        tf.keras.backend.clear_session()
        random.seed(SEED)
        np.random.seed(SEED)
        tf.random.set_seed(SEED)

        model = get_net(input_dim)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
            loss='binary_crossentropy',
            metrics=get_metrics()
        )
        model.fit(X_train, y_train, epochs=EPOCHS, batch_size=bs, verbose=0)
        metrics = model.evaluate(X_test, y_test, verbose=0, return_dict=True)
        return metrics, selected
    except Exception as e:
        print(f"  Error in central training: {e}")
        return None, selected
    finally:
        gc.collect()


# ── Grid search ───────────────────────────────────────────────────────────────
def grid_search(config_name, num_nodes, available_hospitals):
    """
    Run central grid search at a boundary num_nodes.

    Tries all (batch_size, learning_rate) combinations with 3 random iterations.
    Returns (best_lr, best_bs) and saves results CSV for heatmap.
    """
    gs_dir = os.path.join(SCRIPT_DIR, "datasets", "eicu", "data_fixed_rows_results", config_name)
    os.makedirs(gs_dir, exist_ok=True)
    gs_file = os.path.join(gs_dir, f"central_grid_search_n{num_nodes}.csv")

    # Check if grid search was already completed (resumability)
    if os.path.exists(gs_file):
        print(f"\n  Grid search already completed: {gs_file}")
        gs_df = pd.read_csv(gs_file)
        if len(gs_df) > 0:
            best_lr, best_bs = _find_best_params(gs_df)
            print(f"  Loaded best params: LR={best_lr} BS={best_bs}")
            return best_lr, best_bs
        print(f"  File empty, re-running grid search...")

    print(f"\n{'='*60}")
    print(f"CENTRAL GRID SEARCH: config={config_name} num_nodes={num_nodes}")
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
                print(f"\n  [{done}/{total}] BS={bs} LR={lr} Iter={iteration} (n={num_nodes}, hospitals={selected})")

                metrics, _ = run_central_experiment(
                    config_name, num_nodes, iteration, lr, bs, available_hospitals
                )

                if metrics is not None:
                    gs_results.append({
                        'num_nodes': num_nodes,
                        'iteration': iteration,
                        'hospitals': str(selected),
                        'batch_size': bs,
                        'learning_rate': lr,
                        'loss': metrics['loss'],
                        'auc': metrics['auc'],
                        'auprc': metrics['auprc'],
                        'accuracy': metrics['accuracy'],
                        'precision': metrics['precision'],
                        'recall': metrics['recall'],
                    })
                    combo_aucs.append(metrics['auc'])

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


# ── Central runs for a group ──────────────────────────────────────────────────
def run_central_group(config_name, num_nodes_range, lr, bs, available_hospitals):
    """
    Run central experiments for a group of num_nodes using given hyperparameters.
    All 10 iterations are executed. Results are appended to central_results.csv.
    """
    print(f"\n{'='*60}")
    print(f"CENTRAL RUNS: config={config_name} nodes={list(num_nodes_range)} LR={lr} BS={bs}")
    print(f"{'='*60}")

    results_dir = os.path.join(SCRIPT_DIR, "results", "eicu")
    os.makedirs(results_dir, exist_ok=True)
    results_file = os.path.join(results_dir, "central_results.csv")

    # Load or create results dataframe
    if os.path.exists(results_file):
        results_df = pd.read_csv(results_file)
    else:
        results_df = pd.DataFrame(columns=[
            'datetime', 'configuration', 'num_nodes', 'hospitals',
            'iteration', 'lr', 'batch_size', 'epochs',
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
                print(f"  [SKIP] Config={config_name} Nodes={num_nodes} Iter={iteration} (already in results)")
                continue

            selected = select_hospitals(available_hospitals, num_nodes, iteration)
            print(f"\n--- Config={config_name} Nodes={num_nodes} Iter={iteration} LR={lr} BS={bs} Hospitals={selected} ---")

            metrics, _ = run_central_experiment(
                config_name, num_nodes, iteration, lr, bs, available_hospitals
            )

            if metrics is not None:
                new_row = {
                    'datetime': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'configuration': config_name,
                    'num_nodes': num_nodes,
                    'hospitals': str(selected),
                    'iteration': iteration,
                    'lr': lr,
                    'batch_size': bs,
                    'epochs': EPOCHS,
                    'loss': metrics['loss'],
                    'auc': metrics['auc'],
                    'auprc': metrics['auprc'],
                    'accuracy': metrics['accuracy'],
                    'precision': metrics['precision'],
                    'recall': metrics['recall'],
                }
                results_df = pd.concat(
                    [results_df, pd.DataFrame([new_row])], ignore_index=True
                )
                print(f"  -> AUC={metrics['auc']:.4f} AUPRC={metrics['auprc']:.4f}")

            # Save after each num_nodes/iteration for safety
            results_df.to_csv(results_file, index=False)

    print(f"\nCentral results saved to {results_file}")


# ── Main orchestrator ─────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Adaptive central training for eICU with periodic grid search"
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
    print(f"ADAPTIVE CENTRAL TRAINING (eICU) - Started {datetime.now()}")
    print(f"Configurations: {list(configs.keys())}")
    print(f"Grid search: {len(BATCH_SIZES)} batch_sizes x {len(LRS)} lrs x {GRID_SEARCH_ITERS} iterations")
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
            print(f"ERROR: Not enough hospitals ({len(available_hospitals)}) for config {config_name} (need {max_nodes})")
            continue

        # Process groups of GROUP_SIZE
        for group_end in range(GROUP_SIZE, max_nodes + 1, GROUP_SIZE):
            group_start = group_end - GROUP_SIZE + 1
            # First group starts at 2 (no point running central with 1 node)
            if group_start < 2:
                group_start = 2

            print(f"\n--- Group: num_nodes = [{group_start}..{group_end}] ---")

            # Phase 1: Grid search at boundary
            best_lr, best_bs = grid_search(config_name, group_end, available_hospitals)

            # Phase 2: Run central for all nodes in this group
            nodes_range = range(group_start, group_end + 1)
            run_central_group(config_name, nodes_range, best_lr, best_bs, available_hospitals)

    print(f"\n{'#'*60}")
    print(f"ALL EXPERIMENTS COMPLETED - {datetime.now()}")
    print(f"{'#'*60}")


if __name__ == "__main__":
    main()
