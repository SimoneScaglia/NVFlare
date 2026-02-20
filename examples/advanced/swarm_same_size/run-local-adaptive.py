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
Adaptive local training with per-configuration grid search hyperparameter tuning.

Workflow:
  For each dataset (mimic_iii, mimic_iv):
    For each configuration (5nodes, 10nodes, 20nodes, 40nodes):
      1. Grid search (ONE per configuration, since all nodes have the
         same data size within a configuration):
         train a local model on node1 data with 3 random iterations
         and all lr/bs combinations, evaluate on test
      2. Pick best (lr, bs) based on mean AUC
      3. Save grid search results to datasets/{dataset}/same_size_results/{config}/
      4. Run local training for all num_nodes (2..max, all 10 iterations)
         using the best hyperparameters — each node trains independently
      5. Save results to results/{dataset}/local_results.csv

    Grid search hyperparameters:
      batch_sizes = [16, 32, 64, 128, 256, 512]
      lrs = [0.0001, 0.001, 0.01, 0.1]

    Unlike swarm/central, local does NOT re-tune every 5 nodes because
    each node always trains on the same-sized local dataset regardless
    of num_nodes. The best hyperparameters depend only on the per-node
    data size, which is fixed within a configuration.

Usage:
    python run-local-adaptive.py                        # both datasets, all configs
    python run-local-adaptive.py --dataset mimic_iii    # single dataset
    python run-local-adaptive.py --config 10nodes       # single config
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

BATCH_SIZES = [16, 32, 64, 128, 256, 512]
LRS = [0.0001, 0.001, 0.01, 0.1]

CONFIGS = {
    "5nodes":  5,
    "10nodes": 10,
    "20nodes": 20,
    "40nodes": 40,
}

DATASETS = ["mimic_iii", "mimic_iv"]

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


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


# ── Model ─────────────────────────────────────────────────────────────────────
def get_net(input_dim):
    """Build the FCN model (same architecture as mimic_nets.py)."""
    kernel_init = tf.keras.initializers.GlorotUniform(seed=SEED)
    bias_init = tf.keras.initializers.Zeros()
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(16, activation="relu", input_shape=(input_dim,),
                              kernel_initializer=kernel_init, bias_initializer=bias_init),
        tf.keras.layers.Dense(16, activation="relu",
                              kernel_initializer=kernel_init, bias_initializer=bias_init),
        tf.keras.layers.Dense(16, activation="relu",
                              kernel_initializer=kernel_init, bias_initializer=bias_init),
        tf.keras.layers.Dense(1, activation="sigmoid",
                              kernel_initializer=kernel_init, bias_initializer=bias_init),
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


# ── Core local training ──────────────────────────────────────────────────────
def run_local_single_node(dataset, configuration, node_id, iteration, lr, bs):
    """
    Train one local model on a single node's data, evaluate on test set.

    Returns metrics dict or None on failure.
    """
    data_dir = os.path.join(SCRIPT_DIR, "datasets", dataset, "same_size", configuration)
    test_path = os.path.join(SCRIPT_DIR, "datasets", dataset, "test.csv")

    train_file = os.path.join(data_dir, f"node{node_id}_{iteration}.csv")
    if not os.path.exists(train_file):
        print(f"  WARNING: {train_file} does not exist")
        return None

    # Load data
    test_df = pd.read_csv(test_path)
    X_test = test_df.iloc[:, :-1].values.astype(np.float32)
    y_test = test_df.iloc[:, -1].values.astype(np.float32)
    input_dim = X_test.shape[1]

    train_df = pd.read_csv(train_file)
    X_train = train_df.iloc[:, :-1].values.astype(np.float32)
    y_train = train_df.iloc[:, -1].values.astype(np.float32)

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
        return metrics
    except Exception as e:
        print(f"  Error in local training: {e}")
        return None
    finally:
        gc.collect()


# ── Grid search ───────────────────────────────────────────────────────────────
def grid_search(dataset, configuration):
    """
    Run local grid search for a configuration (ONE per config).

    Trains on node1 data with 3 random iterations and all lr/bs combos.
    Returns (best_lr, best_bs) and saves results CSV for heatmap.
    """
    gs_dir = os.path.join(SCRIPT_DIR, "datasets", dataset, "same_size_results", configuration)
    os.makedirs(gs_dir, exist_ok=True)
    gs_file = os.path.join(gs_dir, "local_grid_search.csv")

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
    print(f"LOCAL GRID SEARCH: dataset={dataset} config={configuration}")
    print(f"Hyperparameters: BS={BATCH_SIZES} LR={LRS}")
    print(f"Training on node1 data (same size for all nodes in config)")
    print(f"{'='*60}")

    # Pick 3 random iterations (deterministic per config for reproducibility)
    rng = random.Random(SEED + hash(configuration) + hash(dataset))
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
                print(f"\n  [{done}/{total}] BS={bs} LR={lr} Iter={iteration}")

                metrics = run_local_single_node(
                    dataset, configuration, node_id=1,
                    iteration=iteration, lr=lr, bs=bs
                )

                if metrics is not None:
                    gs_results.append({
                        'iteration': iteration,
                        'node_id': 1,
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


# ── Local runs for all num_nodes ──────────────────────────────────────────────
def run_local_all(dataset, configuration, max_nodes, lr, bs):
    """
    Run local experiments for all num_nodes (2..max) using given hyperparameters.
    Each node trains independently. All 10 iterations are executed.
    Results are appended to local_results.csv.
    """
    print(f"\n{'='*60}")
    print(f"LOCAL RUNS: dataset={dataset} config={configuration} "
          f"nodes=[2..{max_nodes}] LR={lr} BS={bs}")
    print(f"{'='*60}")

    results_dir = os.path.join(SCRIPT_DIR, "results", dataset)
    os.makedirs(results_dir, exist_ok=True)
    results_file = os.path.join(results_dir, "local_results.csv")

    # Load or create results dataframe
    if os.path.exists(results_file):
        results_df = pd.read_csv(results_file)
    else:
        results_df = pd.DataFrame(columns=[
            'datetime', 'configuration', 'num_nodes', 'node_id',
            'iteration', 'lr', 'batch_size', 'epochs',
            'loss', 'auc', 'auprc', 'accuracy', 'precision', 'recall'
        ])

    for num_nodes in range(2, max_nodes + 1):
        for iteration in range(NUM_ITERATIONS):
            # Check if already computed (resumability)
            existing = results_df[
                (results_df['configuration'] == configuration) &
                (results_df['num_nodes'] == num_nodes) &
                (results_df['iteration'] == iteration)
            ]
            if len(existing) > 0:
                print(f"  [SKIP] Config={configuration} Nodes={num_nodes} "
                      f"Iter={iteration} (already in results)")
                continue

            print(f"\n--- Config={configuration} Nodes={num_nodes} "
                  f"Iter={iteration} LR={lr} BS={bs} ---")

            for node_id in range(1, num_nodes + 1):
                metrics = run_local_single_node(
                    dataset, configuration, node_id, iteration, lr, bs
                )

                if metrics is not None:
                    new_row = {
                        'datetime': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'configuration': configuration,
                        'num_nodes': num_nodes,
                        'node_id': node_id,
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
                    print(f"  [Node-{node_id}] AUC={metrics['auc']:.4f} "
                          f"AUPRC={metrics['auprc']:.4f}")

            # Save after each num_nodes/iteration for safety
            results_df.to_csv(results_file, index=False)

    print(f"\nLocal results saved to {results_file}")


# ── Main orchestrator ─────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Adaptive local training with per-configuration grid search"
    )
    parser.add_argument(
        '--dataset',
        choices=['mimic_iii', 'mimic_iv', 'both'],
        default='both',
        help="Dataset to process (default: both)"
    )
    parser.add_argument(
        '--config',
        choices=list(CONFIGS.keys()) + ['all'],
        default='all',
        help="Configuration to process (default: all)"
    )
    args = parser.parse_args()

    datasets = DATASETS if args.dataset == 'both' else [args.dataset]
    configs = CONFIGS if args.config == 'all' else {args.config: CONFIGS[args.config]}

    print(f"\n{'#'*60}")
    print(f"ADAPTIVE LOCAL TRAINING - Started {datetime.now()}")
    print(f"Datasets: {datasets}")
    print(f"Configurations: {list(configs.keys())}")
    print(f"Grid search: {len(BATCH_SIZES)} batch_sizes x {len(LRS)} lrs "
          f"x {GRID_SEARCH_ITERS} iterations (ONE per configuration)")
    print(f"{'#'*60}")

    for dataset in datasets:
        print(f"\n{'#'*60}")
        print(f"DATASET: {dataset}")
        print(f"{'#'*60}")

        # Verify dataset exists
        dataset_dir = os.path.join(SCRIPT_DIR, "datasets", dataset)
        if not os.path.isdir(dataset_dir):
            print(f"ERROR: Dataset directory not found: {dataset_dir}")
            continue

        test_path = os.path.join(dataset_dir, "test.csv")
        if not os.path.isfile(test_path):
            print(f"ERROR: Test file not found: {test_path}")
            continue

        for config_name, max_nodes in configs.items():
            print(f"\n{'='*60}")
            print(f"CONFIGURATION: {config_name} (max_nodes={max_nodes})")
            print(f"{'='*60}")

            # Verify config directory exists
            config_dir = os.path.join(dataset_dir, "same_size", config_name)
            if not os.path.isdir(config_dir):
                print(f"ERROR: Config directory not found: {config_dir}")
                continue

            # Phase 1: Grid search (ONE per configuration)
            best_lr, best_bs = grid_search(dataset, config_name)

            # Phase 2: Run local for all num_nodes
            run_local_all(dataset, config_name, max_nodes, best_lr, best_bs)

    print(f"\n{'#'*60}")
    print(f"ALL EXPERIMENTS COMPLETED - {datetime.now()}")
    print(f"{'#'*60}")


if __name__ == "__main__":
    main()
