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
Fully parallel adaptive local training with per-configuration grid search.

Parallelization strategy (two-phase):
  Phase 1: ALL grid searches across all configs run in parallel
           (each individual experiment is a separate worker task)
  Phase 2: ALL local training experiments run in parallel
           Key optimization: each unique (node_id, iteration) is computed
           ONCE and the result is replicated for all num_nodes that contain
           that node (since local training is independent of num_nodes).

Results are collected in the main process and saved sorted by
(configuration, num_nodes, node_id, iteration) — identical to sequential output.

Usage:
    python run-local-adaptive.py                        # both datasets, all configs
    python run-local-adaptive.py --dataset mimic_iii    # single dataset
    python run-local-adaptive.py --config 10nodes       # single config
    python run-local-adaptive.py --workers 32           # set parallelism
"""

import argparse
import gc
import multiprocessing as mp
import os
import random
import warnings
from datetime import datetime

import numpy as np
import pandas as pd

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

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
    "80nodes": 80,
}

DATASETS = ["mimic_iii", "mimic_iv", "mimic_iv_fixed"]

DEFAULT_WORKERS = max(1, min(32, (os.cpu_count() or 1) - 1))
PARALLEL_WORKERS = int(os.environ.get("NVFLARE_LOCAL_WORKERS", DEFAULT_WORKERS))

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
tf.get_logger().setLevel("ERROR")
warnings.filterwarnings(
    "ignore",
    message=r"Do not pass an `input_shape`/`input_dim` argument to a layer\.",
    category=UserWarning,
    module=r"keras\.src\.layers\.core\.dense",
)


def _init_worker():
    """Initialize each spawned worker process."""
    set_seed()
    tf.get_logger().setLevel("ERROR")
    warnings.filterwarnings("ignore")


# ── Model ─────────────────────────────────────────────────────────────────────
def get_net(input_dim):
    """Build the FCN model (same architecture as mimic_nets.py)."""
    kernel_init = tf.keras.initializers.GlorotUniform(seed=SEED)
    bias_init = tf.keras.initializers.Zeros()
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(16, activation="relu",
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


# ── Worker function ───────────────────────────────────────────────────────────
def _local_experiment_worker(args):
    """Pool worker: runs one local training experiment."""
    dataset, configuration, node_id, iteration, lr, bs = args
    set_seed()
    return run_local_single_node(dataset, configuration, node_id, iteration, lr, bs)


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
        return None

    try:
        test_df = pd.read_csv(test_path)
        X_test = test_df.iloc[:, :-1].values.astype(np.float32)
        y_test = test_df.iloc[:, -1].values.astype(np.float32)
        input_dim = X_test.shape[1]

        train_df = pd.read_csv(train_file)
        X_train = train_df.iloc[:, :-1].values.astype(np.float32)
        y_train = train_df.iloc[:, -1].values.astype(np.float32)

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
        print(f"  Error local training (node={node_id} iter={iteration}): {e}")
        return None
    finally:
        tf.keras.backend.clear_session()
        gc.collect()


# ── Helpers ───────────────────────────────────────────────────────────────────
def _find_best_params(gs_df):
    """Find best (lr, bs) from grid search results based on mean AUC."""
    grouped = gs_df.groupby(['batch_size', 'learning_rate'])['auc'].mean()
    best_idx = grouped.idxmax()
    best_bs, best_lr = best_idx
    best_auc = grouped[best_idx]
    print(f"  Best combo: LR={best_lr} BS={best_bs} (mean AUC={best_auc:.4f})")
    return best_lr, int(best_bs)


# ── Main orchestrator ─────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Fully parallel adaptive local training"
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
    parser.add_argument(
        '--workers', type=int, default=PARALLEL_WORKERS,
        help=f"Number of parallel workers (default: {PARALLEL_WORKERS}, "
             f"env: NVFLARE_LOCAL_WORKERS)"
    )
    args = parser.parse_args()

    datasets = DATASETS if args.dataset == 'both' else [args.dataset]
    configs = CONFIGS if args.config == 'all' else {args.config: CONFIGS[args.config]}
    workers = args.workers

    print(f"\n{'#'*60}")
    print(f"PARALLEL ADAPTIVE LOCAL TRAINING - Started {datetime.now()}")
    print(f"Datasets: {datasets}")
    print(f"Configurations: {list(configs.keys())}")
    print(f"Workers: {workers}")
    print(f"Grid search: {len(BATCH_SIZES)} BS x {len(LRS)} LR "
          f"x {GRID_SEARCH_ITERS} iters (ONE per config)")
    print(f"{'#'*60}")

    for dataset in datasets:
        print(f"\n{'#'*60}")
        print(f"DATASET: {dataset}")
        print(f"{'#'*60}")

        dataset_dir = os.path.join(SCRIPT_DIR, "datasets", dataset)
        if not os.path.isdir(dataset_dir):
            print(f"ERROR: Dataset directory not found: {dataset_dir}")
            continue
        test_path = os.path.join(dataset_dir, "test.csv")
        if not os.path.isfile(test_path):
            print(f"ERROR: Test file not found: {test_path}")
            continue

        # ── PHASE 1: All Grid Searches in Parallel ────────────────────────
        print(f"\n{'='*60}")
        print(f"PHASE 1: Grid Searches (all configs in parallel)")
        print(f"{'='*60}")

        gs_tasks = []    # (dataset, config, 1, iteration, lr, bs)
        gs_meta = []     # (config, iteration, bs, lr)
        gs_loaded = {}   # config -> already-done DataFrame

        for config_name, max_nodes in configs.items():
            config_dir = os.path.join(dataset_dir, "same_size", config_name)
            if not os.path.isdir(config_dir):
                print(f"  WARNING: Config dir not found: {config_dir}")
                continue

            gs_dir = os.path.join(
                SCRIPT_DIR, "datasets", dataset,
                "same_size_results", config_name
            )
            os.makedirs(gs_dir, exist_ok=True)
            gs_file = os.path.join(gs_dir, "local_grid_search.csv")

            # Check if already completed (resumability)
            if os.path.exists(gs_file):
                gs_df = pd.read_csv(gs_file)
                if len(gs_df) > 0:
                    gs_loaded[config_name] = gs_df
                    print(f"  [LOADED] Grid search: {config_name}")
                    continue

            # Need to compute
            rng = random.Random(
                SEED + hash(config_name) + hash(dataset)
            )
            iterations = sorted(
                rng.sample(range(NUM_ITERATIONS), GRID_SEARCH_ITERS)
            )
            print(f"  Grid search needed: {config_name} "
                  f"(iterations: {iterations})")

            for bs in BATCH_SIZES:
                for lr in LRS:
                    for iteration in iterations:
                        gs_tasks.append((
                            dataset, config_name, 1,
                            iteration, lr, bs
                        ))
                        gs_meta.append((
                            config_name, iteration, bs, lr
                        ))

        # Execute grid search tasks in parallel
        if gs_tasks:
            print(f"\n  Submitting {len(gs_tasks)} grid search experiments "
                  f"to {workers} workers...")
            ctx = mp.get_context("spawn")
            with ctx.Pool(
                processes=workers, initializer=_init_worker
            ) as pool:
                gs_results = pool.map(
                    _local_experiment_worker, gs_tasks, chunksize=1
                )

            # Organize by config
            gs_data = {}
            for meta, metrics in zip(gs_meta, gs_results):
                config_name, iteration, bs, lr = meta
                if config_name not in gs_data:
                    gs_data[config_name] = []
                if metrics is not None:
                    gs_data[config_name].append({
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

            # Save grid search CSVs
            for config_name, results in gs_data.items():
                gs_dir = os.path.join(
                    SCRIPT_DIR, "datasets", dataset,
                    "same_size_results", config_name
                )
                gs_file = os.path.join(gs_dir, "local_grid_search.csv")
                gs_df = pd.DataFrame(results)
                if len(gs_df) > 0:
                    gs_df = gs_df.sort_values(
                        by=['batch_size', 'learning_rate',
                            'iteration', 'node_id']
                    )
                gs_df.to_csv(gs_file, index=False)
                gs_loaded[config_name] = gs_df
                print(f"  [SAVED] Grid search: {config_name} "
                      f"({len(results)} results)")
        else:
            print("  All grid searches already completed.")

        # Find best params for each config
        best_params = {}
        for config_name, gs_df in gs_loaded.items():
            best_lr, best_bs = _find_best_params(gs_df)
            best_params[config_name] = (best_lr, best_bs)
            print(f"  BEST for {config_name}: LR={best_lr} BS={best_bs}")

        # ── PHASE 2: All Local Runs in Parallel ──────────────────────────
        print(f"\n{'='*60}")
        print(f"PHASE 2: Local Runs (all configs/nodes/iters in parallel)")
        print(f"Optimization: each unique (node_id, iteration) computed once")
        print(f"{'='*60}")

        results_dir = os.path.join(SCRIPT_DIR, "results", dataset)
        os.makedirs(results_dir, exist_ok=True)
        results_file = os.path.join(results_dir, "local_results.csv")

        if os.path.exists(results_file):
            existing_df = pd.read_csv(results_file)
        else:
            existing_df = pd.DataFrame(columns=[
                'datetime', 'configuration', 'num_nodes', 'node_id',
                'iteration', 'lr', 'batch_size', 'epochs',
                'loss', 'auc', 'auprc', 'accuracy',
                'precision', 'recall'
            ])

        for config_name, max_nodes in configs.items():
            config_dir = os.path.join(dataset_dir, "same_size", config_name)
            if not os.path.isdir(config_dir):
                continue
            if config_name not in best_params:
                print(f"  WARNING: No best params for {config_name}")
                continue

            lr, bs = best_params[config_name]

            # Find which (num_nodes, iteration) pairs need computation
            needed_pairs = []
            for num_nodes in range(2, max_nodes + 1):
                for iteration in range(NUM_ITERATIONS):
                    mask = (
                        (existing_df['configuration'] == config_name)
                        & (existing_df['num_nodes'] == num_nodes)
                        & (existing_df['iteration'] == iteration)
                    )
                    if not mask.any():
                        needed_pairs.append((num_nodes, iteration))

            if not needed_pairs:
                print(f"  [SKIP] {config_name}: all experiments done")
                continue

            # Find unique (node_id, iteration) that need computation
            # Key insight: local training for node N with iteration I
            # produces the same result regardless of num_nodes
            unique_needed = set()
            for num_nodes, iteration in needed_pairs:
                for node_id in range(1, num_nodes + 1):
                    unique_needed.add((node_id, iteration))

            unique_tasks = [
                (dataset, config_name, node_id, iteration, lr, bs)
                for node_id, iteration in sorted(unique_needed)
            ]

            print(f"\n  {config_name}: {len(needed_pairs)} (num_nodes, iter) "
                  f"pairs → {len(unique_needed)} unique experiments "
                  f"(vs {sum(nn for nn, _ in needed_pairs)} naive)")

            # Execute unique experiments in parallel
            ctx = mp.get_context("spawn")
            with ctx.Pool(
                processes=workers, initializer=_init_worker
            ) as pool:
                unique_results = pool.map(
                    _local_experiment_worker, unique_tasks, chunksize=1
                )

            # Build results map: (node_id, iteration) -> metrics
            results_map = {}
            for task, metrics in zip(unique_tasks, unique_results):
                _, _, node_id, iteration, _, _ = task
                results_map[(node_id, iteration)] = metrics

            # Build complete result rows for all (num_nodes, iteration)
            new_rows = []
            for num_nodes, iteration in sorted(needed_pairs):
                for node_id in range(1, num_nodes + 1):
                    metrics = results_map.get((node_id, iteration))
                    if metrics is not None:
                        new_rows.append({
                            'datetime': datetime.now().strftime(
                                '%Y-%m-%d %H:%M:%S'
                            ),
                            'configuration': config_name,
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
                        })

            if new_rows:
                new_df = pd.DataFrame(new_rows)
                existing_df = pd.concat(
                    [existing_df, new_df], ignore_index=True
                )
                # Sort to match sequential output order
                existing_df = existing_df.sort_values(
                    by=['configuration', 'num_nodes',
                        'node_id', 'iteration']
                ).reset_index(drop=True)
                existing_df.to_csv(results_file, index=False)
                print(f"  [SAVED] {config_name}: {len(new_rows)} rows → "
                      f"{results_file}")

    print(f"\n{'#'*60}")
    print(f"ALL EXPERIMENTS COMPLETED - {datetime.now()}")
    print(f"{'#'*60}")


if __name__ == "__main__":
    main()
