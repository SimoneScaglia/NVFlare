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
Fully parallel adaptive local training for eICU dataset with per-configuration
grid search.

Key differences from MIMIC version:
  - Data is per-hospital: {hospitalid}_train.csv and {hospitalid}_test.csv
  - Each iteration randomly selects num_nodes hospitals from the available pool
  - Each hospital trains independently on its own train, evaluates on its own test
  - Hospital selection is deterministic per iteration (for reproducibility),
    and smaller num_nodes selections are always subsets of larger ones

Parallelization strategy (two-phase):
  Phase 1: ALL grid searches across all configs run in parallel
           (each individual experiment is a separate worker task)
  Phase 2: ALL local training experiments run in parallel
           Key optimization: each unique hospital_id is computed ONCE and the
           result is replicated for all (num_nodes, iteration) pairs that
           include that hospital (since local training is independent of
           num_nodes).

Workflow:
  For each configuration (5nodes, 10nodes, 20nodes, 25nodes):
    1. Grid search (ONE per configuration, since all hospitals have the
       same data size within a configuration):
       train a local model on a random hospital's data with 3 random iterations
       and all lr/bs combinations, evaluate on that hospital's test
    2. Pick best (lr, bs) based on mean AUC
    3. Save grid search results
    4. Run local training for all num_nodes (2..max, all 10 iterations)
       using the best hyperparameters — each hospital trains independently
    5. Save results to results/eicu/local_results.csv

    Grid search hyperparameters:
      batch_sizes = [16, 32, 64, 128, 256, 512]
      lrs = [0.0001, 0.001, 0.01, 0.1]

    Unlike swarm/central, local does NOT re-tune every 5 nodes because
    each hospital always trains on the same-sized local dataset regardless
    of num_nodes.

Checkpointing / resumability:
  - Grid search results are saved row-by-row; interrupted runs resume from
    the last completed (batch_size, learning_rate, iteration) triple.
  - Local results are saved after each (num_nodes, iteration); the CSV is
    reloaded on restart and completed entries are skipped automatically.
  - Use --resume-config / --resume-num-nodes / --resume-iteration to skip
    directly to a known restart point without replaying the CSV-check loop.

Usage:
    python eicu_run-local-adaptive.py                                    # all configs
    python eicu_run-local-adaptive.py --config 10nodes                   # single config
    python eicu_run-local-adaptive.py --workers 32                       # set parallelism
    python eicu_run-local-adaptive.py --resume-config 20nodes            # resume a config
    python eicu_run-local-adaptive.py --resume-config 20nodes \\
        --resume-num-nodes 12 --resume-iteration 3                       # precise resume
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
    "25nodes": 25,
}

DEFAULT_WORKERS = max(1, min(32, (os.cpu_count() or 1) - 1))
PARALLEL_WORKERS = int(os.environ.get("NVFLARE_LOCAL_WORKERS", DEFAULT_WORKERS))

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


# ── Core local training ──────────────────────────────────────────────────────
def run_local_single_hospital(config_name, hospital_id, lr, bs):
    """
    Train one local model on a single hospital's data, evaluate on its test.

    Returns metrics dict or None on failure.
    """
    data_dir = os.path.join(EICU_DATA_DIR, config_name)

    train_file = os.path.join(data_dir, f"{hospital_id}_train.csv")
    test_file = os.path.join(data_dir, f"{hospital_id}_test.csv")

    if not os.path.exists(train_file) or not os.path.exists(test_file):
        print(f"  WARNING: Files for hospital {hospital_id} not found")
        return None

    # Load data
    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)

    X_train = train_df.iloc[:, :-1].values.astype(np.float32)
    y_train = train_df.iloc[:, -1].values.astype(np.float32)
    X_test = test_df.iloc[:, :-1].values.astype(np.float32)
    y_test = test_df.iloc[:, -1].values.astype(np.float32)
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
        return metrics
    except Exception as e:
        print(f"  Error in local training: {e}")
        return None
    finally:
        gc.collect()


# ── Worker function ───────────────────────────────────────────────────────────
def _local_experiment_worker(args):
    """Pool worker: runs one local training experiment."""
    config_name, hospital_id, lr, bs = args
    set_seed()
    return run_local_single_hospital(config_name, hospital_id, lr, bs)


# ── Grid search ───────────────────────────────────────────────────────────────
def grid_search(config_name, available_hospitals, workers=1):
    """
    Run local grid search for a configuration (ONE per config).

    Trains on a random hospital's data with 3 random iterations
    (different hospital each iteration) and all lr/bs combos.
    Returns (best_lr, best_bs) and saves results CSV for heatmap.

    Checkpointing: results are saved incrementally; an interrupted grid
    search is resumed from the last completed (bs, lr, iteration) triple.
    Tasks are executed in parallel using a multiprocessing pool.
    """
    gs_dir = os.path.join(SCRIPT_DIR, "datasets", "eicu", "data_fixed_rows_results", config_name)
    os.makedirs(gs_dir, exist_ok=True)
    gs_file = os.path.join(gs_dir, "local_grid_search.csv")

    # Pick 3 random iterations and corresponding hospitals
    rng = random.Random(SEED + hash(config_name))
    iterations = sorted(rng.sample(range(NUM_ITERATIONS), GRID_SEARCH_ITERS))
    gs_hospitals = []
    for it in iterations:
        selected = select_hospitals(available_hospitals, 1, it)
        gs_hospitals.append(selected[0])
    all_combos = {(bs, lr, it) for bs in BATCH_SIZES for lr in LRS for it in iterations}

    # Load existing partial or complete results
    if os.path.exists(gs_file):
        gs_df_existing = pd.read_csv(gs_file)
    else:
        gs_df_existing = pd.DataFrame()

    if len(gs_df_existing) > 0:
        done_combos = set(
            zip(
                gs_df_existing['batch_size'],
                gs_df_existing['learning_rate'],
                gs_df_existing['iteration'],
            )
        )
        if all_combos.issubset(done_combos):
            print(f"\n  Grid search already completed: {gs_file}")
            best_lr, best_bs = _find_best_params(gs_df_existing)
            print(f"  Loaded best params: LR={best_lr} BS={best_bs}")
            return best_lr, best_bs
        remaining = len(all_combos - done_combos)
        print(f"\n  Grid search partially done "
              f"({len(done_combos)}/{len(all_combos)} combos). "
              f"Resuming {remaining} remaining...")
        gs_results = gs_df_existing.to_dict('records')
    else:
        done_combos = set()
        gs_results = []

    print(f"\n{'='*60}")
    print(f"LOCAL GRID SEARCH (eICU): config={config_name}")
    print(f"Hyperparameters: BS={BATCH_SIZES} LR={LRS}")
    print(f"{'='*60}")
    print(f"Selected iterations for grid search: {iterations}")
    print(f"Corresponding hospitals: {gs_hospitals}")

    # Build list of remaining tasks
    gs_tasks = []   # (config_name, hid, lr, bs)
    gs_meta = []    # (bs, lr, iteration, hid)
    for bs in BATCH_SIZES:
        for lr in LRS:
            for idx, iteration in enumerate(iterations):
                if (bs, lr, iteration) in done_combos:
                    continue
                hid = gs_hospitals[idx]
                gs_tasks.append((config_name, hid, lr, bs))
                gs_meta.append((bs, lr, iteration, hid))

    if gs_tasks:
        print(f"\n  Submitting {len(gs_tasks)} grid search experiments "
              f"to {workers} workers...")

        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=workers, initializer=_init_worker) as pool:
            gs_metrics = pool.map(
                _local_experiment_worker, gs_tasks, chunksize=1
            )

        # Collect results
        for meta, metrics in zip(gs_meta, gs_metrics):
            bs, lr, iteration, hid = meta
            if metrics is not None:
                gs_results.append({
                    'iteration': iteration,
                    'hospital_id': hid,
                    'batch_size': bs,
                    'learning_rate': lr,
                    'loss': metrics['loss'],
                    'auc': metrics['auc'],
                    'auprc': metrics['auprc'],
                    'accuracy': metrics['accuracy'],
                    'precision': metrics['precision'],
                    'recall': metrics['recall'],
                })

    # Save grid search results
    gs_df = pd.DataFrame(gs_results)
    if len(gs_df) > 0:
        gs_df = gs_df.sort_values(
            by=['batch_size', 'learning_rate', 'iteration']
        )
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
def run_local_all(
    config_name, max_nodes, lr, bs, available_hospitals,
    workers=1, resume_num_nodes=None, resume_iteration=None
):
    """
    Run local experiments for all num_nodes (2..max) using given hyperparameters.
    Each hospital trains independently on its own data.
    All 10 iterations are executed. Results are appended to local_results.csv.

    Parallelization: each unique hospital_id is trained ONCE and the result is
    replicated for all (num_nodes, iteration) pairs that include it.

    resume_num_nodes: skip num_nodes strictly below this value.
    resume_iteration: for the first non-skipped num_nodes, skip iterations
                      strictly below this value.
    """
    print(f"\n{'='*60}")
    print(f"LOCAL RUNS (eICU): config={config_name} "
          f"nodes=[2..{max_nodes}] LR={lr} BS={bs}")
    print(f"Optimization: each unique hospital_id computed once")
    print(f"{'='*60}")

    results_dir = os.path.join(SCRIPT_DIR, "results", "eicu")
    os.makedirs(results_dir, exist_ok=True)
    results_file = os.path.join(results_dir, "local_results.csv")

    # Load or create results dataframe
    if os.path.exists(results_file):
        existing_df = pd.read_csv(results_file)
    else:
        existing_df = pd.DataFrame(columns=[
            'datetime', 'configuration', 'num_nodes', 'hospital_id',
            'iteration', 'lr', 'batch_size', 'epochs',
            'loss', 'auc', 'auprc', 'accuracy', 'precision', 'recall'
        ])

    # Find which (num_nodes, iteration) pairs need computation
    resume_num_nodes_crossed = False
    needed_pairs = []

    for num_nodes in range(2, max_nodes + 1):
        # ── Explicit resume-point skip ────────────────────────────────────
        if resume_num_nodes is not None and num_nodes < resume_num_nodes:
            continue

        # Determine iteration start for this num_nodes
        if (
            not resume_num_nodes_crossed
            and resume_num_nodes is not None
            and num_nodes == resume_num_nodes
            and resume_iteration is not None
        ):
            iter_start = resume_iteration
        else:
            iter_start = 0
        resume_num_nodes_crossed = True

        for iteration in range(iter_start, NUM_ITERATIONS):
            # Check if already computed (CSV-based resumability)
            mask = (
                (existing_df['configuration'] == config_name)
                & (existing_df['num_nodes'] == num_nodes)
                & (existing_df['iteration'] == iteration)
            )
            if mask.any():
                continue
            needed_pairs.append((num_nodes, iteration))

    if not needed_pairs:
        print(f"  [SKIP] {config_name}: all experiments done")
        return

    # Find unique hospital_ids that need computation.
    # Key insight: local training for a given hospital_id always produces
    # the same result regardless of (num_nodes, iteration), since lr/bs
    # are fixed within a config.
    unique_hospitals = set()
    for num_nodes, iteration in needed_pairs:
        selected = select_hospitals(available_hospitals, num_nodes, iteration)
        for hid in selected:
            unique_hospitals.add(hid)

    unique_tasks = [
        (config_name, hid, lr, bs)
        for hid in sorted(unique_hospitals)
    ]

    naive_count = sum(nn for nn, _ in needed_pairs)
    print(f"\n  {config_name}: {len(needed_pairs)} (num_nodes, iter) "
          f"pairs → {len(unique_hospitals)} unique hospital experiments "
          f"(vs {naive_count} naive)")

    # Execute unique experiments in parallel
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=workers, initializer=_init_worker) as pool:
        unique_results = pool.map(
            _local_experiment_worker, unique_tasks, chunksize=1
        )

    # Build results map: hospital_id -> metrics
    results_map = {}
    for task, metrics in zip(unique_tasks, unique_results):
        _, hid, _, _ = task
        results_map[hid] = metrics

    # Build complete result rows for all (num_nodes, iteration)
    new_rows = []
    for num_nodes, iteration in sorted(needed_pairs):
        selected = select_hospitals(available_hospitals, num_nodes, iteration)
        for hid in selected:
            metrics = results_map.get(hid)
            if metrics is not None:
                new_rows.append({
                    'datetime': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'configuration': config_name,
                    'num_nodes': num_nodes,
                    'hospital_id': hid,
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
            by=['configuration', 'num_nodes', 'hospital_id', 'iteration']
        ).reset_index(drop=True)
        existing_df.to_csv(results_file, index=False)
        print(f"  [SAVED] {config_name}: {len(new_rows)} rows → "
              f"{results_file}")

    print(f"\nLocal results saved to {results_file}")


# ── Main orchestrator ─────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Fully parallel adaptive local training for eICU "
                    "with per-configuration grid search"
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
    # ── Explicit resume parameters ──────────────────────────────────────
    parser.add_argument(
        '--resume-config',
        choices=list(CONFIGS.keys()),
        default=None,
        metavar='CONFIG',
        help="Resume from this configuration, skipping earlier ones "
             "(e.g. 20nodes)"
    )
    parser.add_argument(
        '--resume-num-nodes',
        type=int,
        default=None,
        metavar='N',
        help="Within the resumed config, skip num_nodes values below N "
             "(requires --resume-config)"
    )
    parser.add_argument(
        '--resume-iteration',
        type=int,
        default=None,
        metavar='I',
        help="Within the resumed (config, num_nodes), skip iterations below I "
             "(requires --resume-config and --resume-num-nodes)"
    )
    args = parser.parse_args()

    # ── Validate resume args ─────────────────────────────────────────────
    if args.resume_num_nodes is not None and args.resume_config is None:
        parser.error("--resume-num-nodes requires --resume-config")
    if args.resume_iteration is not None and args.resume_num_nodes is None:
        parser.error("--resume-iteration requires --resume-num-nodes")

    configs = CONFIGS if args.config == 'all' else {args.config: CONFIGS[args.config]}
    workers = args.workers

    print(f"\n{'#'*60}")
    print(f"PARALLEL ADAPTIVE LOCAL TRAINING (eICU) - Started {datetime.now()}")
    print(f"Configurations: {list(configs.keys())}")
    print(f"Workers: {workers}")
    print(f"Grid search: {len(BATCH_SIZES)} batch_sizes x {len(LRS)} lrs "
          f"x {GRID_SEARCH_ITERS} iterations (ONE per configuration)")
    if args.resume_config:
        print(f"Resume point: config={args.resume_config} "
              f"num_nodes={args.resume_num_nodes} "
              f"iteration={args.resume_iteration}")
    print(f"{'#'*60}")

    # Build ordered list of config names to know which ones precede the resume point
    all_config_names = list(CONFIGS.keys())

    for config_name, max_nodes in configs.items():
        # ── Config-level resume skip ─────────────────────────────────────
        if args.resume_config is not None:
            resume_idx = all_config_names.index(args.resume_config)
            config_idx = all_config_names.index(config_name)
            if config_idx < resume_idx:
                print(f"\n[RESUME SKIP] Configuration {config_name} "
                      f"(before resume config {args.resume_config})")
                continue

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

        # Determine per-config resume parameters
        is_resume_config = (args.resume_config == config_name)
        cfg_resume_num_nodes = args.resume_num_nodes if is_resume_config else None
        cfg_resume_iteration = args.resume_iteration if is_resume_config else None

        # Phase 1: Grid search (ONE per configuration)
        best_lr, best_bs = grid_search(config_name, available_hospitals, workers=workers)

        # Phase 2: Run local for all num_nodes
        run_local_all(
            config_name, max_nodes, best_lr, best_bs, available_hospitals,
            workers=workers,
            resume_num_nodes=cfg_resume_num_nodes,
            resume_iteration=cfg_resume_iteration,
        )

    print(f"\n{'#'*60}")
    print(f"ALL EXPERIMENTS COMPLETED - {datetime.now()}")
    print(f"{'#'*60}")


if __name__ == "__main__":
    main()
