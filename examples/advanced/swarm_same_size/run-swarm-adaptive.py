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
Fully parallel adaptive swarm learning with periodic grid search.

Parallelization strategy (two-phase, memory-safe):
  Phase 1: Grid searches processed sequentially per boundary group,
           with dynamic worker count based on num_nodes.
  Phase 2: Runs batched by num_nodes with adaptive parallelism.

OOM prevention (tuned for n2-standard-32, 128 GB RAM):
  - Dynamic workers: fewer parallel processes for heavier simulations
  - Sequential boundary processing: only one group_end at a time
  - maxtasksperchild=1: worker process recycled after each task
  - imap_unordered: results streamed incrementally
  - periodic saves: results flushed to disk every N experiments
  - gc.collect() between batches

Each swarm experiment is fully isolated:
  - Unique data dir:  /tmp/data/{uuid}/
  - Unique job name:  p_{uuid}
  - Unique workspace: /tmp/nvflare/p_{uuid}/

Uses parallel_swarm_cse_tf_model_learner template to avoid conflicts
with running eicu processes.

Usage:
    python run-swarm-adaptive.py                         # both datasets, all configs
    python run-swarm-adaptive.py --dataset mimic_iii     # single dataset
    python run-swarm-adaptive.py --config 10nodes        # single config
    python run-swarm-adaptive.py --workers 8             # max parallelism cap
"""

import argparse
import gc
import multiprocessing as mp
import os
import random
import re
import shutil
import subprocess
import sys
import uuid
import warnings
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
LRS = [0.001, 0.01, 0.1]

CONFIGS = {
    "80nodes": 80,
}

DATASETS = ["mimic_iii", "mimic_iv", "mimic_iv_fixed"]

# Conservative default: each NVFlare simulator is heavy (multi-threaded)
# Each simulator spawns num_nodes threads, so keep parallelism moderate.
# On n2-standard-32 (128 GB RAM), max_workers is a cap; actual workers
# are scaled down dynamically based on num_nodes per experiment.
DEFAULT_WORKERS = 4
PARALLEL_WORKERS = int(os.environ.get("NVFLARE_SWARM_WORKERS", DEFAULT_WORKERS))

# Memory budget: ~128 GB total, ~110 GB usable.
# Rough per-worker estimate: base ~1.5 GB + ~0.02 GB per node in simulator.
TOTAL_RAM_GB = 110
BASE_RAM_PER_WORKER_GB = 1.5
RAM_PER_NODE_GB = 0.02

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SWARM_TF_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "swarm_learning_tf"))

# Template name for parallel-safe jobs
PARALLEL_TEMPLATE = "parallel_swarm_cse_tf_model_learner"

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


def _init_worker():
    """Initialize each spawned worker process."""
    set_seed()
    tf.get_logger().setLevel("ERROR")
    warnings.filterwarnings("ignore")


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
            print(f"  STDOUT (last 500): {result.stdout[-500:]}")
        if result.stderr:
            print(f"  STDERR (last 500): {result.stderr[-500:]}")
    return result


# ── Job config editing (per-job, not template) ────────────────────────────────
def edit_job_config(job_dir, train_idx_root, min_responses_required,
                    learning_rate, batch_size):
    """Edit a created job's config_fed_client.conf (NOT the template)."""
    config_path = os.path.join(
        job_dir, "app", "config", "config_fed_client.conf"
    )
    with open(config_path, 'r') as f:
        content = f.read()

    content = re.sub(
        r'train_idx_root\s*=\s*"[^"]*"',
        f'train_idx_root = "{train_idx_root}"',
        content
    )
    content = re.sub(
        r'min_responses_required\s*=\s*\d+',
        f'min_responses_required = {min_responses_required}',
        content
    )

    learner_pattern = r'(id = "mimic-learner"[^{]*args \{)([^}]+)(\})'

    def update_learner_args(match):
        args_section = match.group(2)
        args_section = re.sub(
            r'lr\s*=\s*[\d.eE+-]+', f'lr = {learning_rate}', args_section
        )
        args_section = re.sub(
            r'batch_size\s*=\s*\d+', f'batch_size = {batch_size}', args_section
        )
        return match.group(1) + args_section + match.group(3)

    content = re.sub(
        learner_pattern, update_learner_args, content, flags=re.DOTALL
    )

    with open(config_path, 'w') as f:
        f.write(content)


# ── Data file management ─────────────────────────────────────────────────────
def copy_data_files_isolated(data_dir, num_nodes, iteration, dest_dir):
    """Copy node CSV files to an isolated temp dir as site-{i}.csv."""
    os.makedirs(dest_dir, exist_ok=True)
    for i in range(1, num_nodes + 1):
        src = os.path.join(data_dir, f"node{i}_{iteration}.csv")
        dst = os.path.join(dest_dir, f"site-{i}.csv")
        if os.path.isfile(src):
            shutil.copy(src, dst)
        else:
            print(f"  WARNING: {src} does not exist!")


# ── Metrics helpers ───────────────────────────────────────────────────────────
def get_metrics_fns():
    return [
        tf.keras.metrics.AUC(name='auc', curve='ROC', num_thresholds=1000),
        tf.keras.metrics.AUC(name='auprc', curve='PR', num_thresholds=1000),
        tf.keras.metrics.BinaryAccuracy(name='accuracy'),
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall'),
    ]


# ── Core isolated swarm experiment ────────────────────────────────────────────
def _run_swarm_isolated(dataset, configuration, num_nodes, iteration, lr, bs):
    """
    Run one fully isolated swarm experiment.

    Creates unique temp dirs, runs simulator, evaluates, cleans up.
    Returns list of (site_id, metrics_dict) tuples.
    """
    set_seed()

    unique_id = f"p_{uuid.uuid4().hex[:12]}"
    data_dir = os.path.join(
        SCRIPT_DIR, "datasets", dataset, "same_size", configuration
    )
    test_path = os.path.join(SCRIPT_DIR, "datasets", dataset, "test.csv")

    tmp_data_dir = f"/tmp/data/{unique_id}"
    job_name = unique_id
    job_dir = os.path.join(SWARM_TF_DIR, "jobs", job_name)
    workspace = f"/tmp/nvflare/{unique_id}"

    try:
        # 1. Copy data files to isolated temp dir
        copy_data_files_isolated(data_dir, num_nodes, iteration, tmp_data_dir)

        # 2. Create NVFlare job from parallel template
        r = run_command(
            f"nvflare job create -j ./jobs/{job_name} "
            f"-w {PARALLEL_TEMPLATE} -sd ./code -force",
            cwd=SWARM_TF_DIR
        )
        if r.returncode != 0:
            print(f"  [{unique_id}] Job creation failed")
            return []

        # 3. Edit the created job's config (NOT the template)
        edit_job_config(job_dir, tmp_data_dir, num_nodes, lr, bs)

        # 4. Run NVFlare simulator with isolated workspace
        r = run_command(
            f"nvflare simulator ./jobs/{job_name} "
            f"-w {workspace} -n {num_nodes} -t {num_nodes}",
            cwd=SWARM_TF_DIR
        )
        if r.returncode != 0:
            print(f"  [{unique_id}] Simulator failed "
                  f"(n={num_nodes} iter={iteration})")
            return []

        # 5. Evaluate global models
        test_df = pd.read_csv(test_path)
        X_test = test_df.iloc[:, :-1].values.astype(np.float32)
        y_test = test_df.iloc[:, -1].values.astype(np.float32)
        input_dim = X_test.shape[1]

        site_results = []
        for j in range(1, num_nodes + 1):
            model_path = os.path.join(
                workspace, f'site-{j}', 'simulate_job',
                f'app_site-{j}', f'site-{j}.weights.h5'
            )
            if not os.path.exists(model_path):
                print(f"  [{unique_id}] Model not found: site-{j}")
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
                metrics = model.evaluate(
                    X_test, y_test, verbose=0, return_dict=True
                )
                site_results.append((j, metrics))
            except Exception as e:
                print(f"  [{unique_id}] Error evaluating site-{j}: {e}")
            finally:
                gc.collect()

        return site_results

    except Exception as e:
        print(f"  [{unique_id}] Unexpected error: {e}")
        return []

    finally:
        # 6. Clean up all isolated paths
        shutil.rmtree(tmp_data_dir, ignore_errors=True)
        shutil.rmtree(workspace, ignore_errors=True)
        shutil.rmtree(job_dir, ignore_errors=True)


# ── Pool workers (return metadata with results for imap_unordered) ────────────
def _gs_worker(packed_args):
    """Grid search worker: unpacks metadata + experiment args, returns both."""
    meta, experiment_args = packed_args
    site_results = _run_swarm_isolated(*experiment_args)
    return (meta, site_results)


def _run_worker(packed_args):
    """Run worker: unpacks metadata + experiment args, returns both."""
    meta, experiment_args = packed_args
    site_results = _run_swarm_isolated(*experiment_args)
    return (meta, site_results)


# ── Dynamic worker scaling ─────────────────────────────────────────────────────
def _max_workers_for(num_nodes, cap):
    """Compute safe number of parallel workers for a given num_nodes.

    Each NVFlare simulator with N nodes uses approximately:
        BASE_RAM_PER_WORKER_GB + RAM_PER_NODE_GB * N  (GB)
    We aim to stay well within TOTAL_RAM_GB.
    """
    per_worker = BASE_RAM_PER_WORKER_GB + RAM_PER_NODE_GB * num_nodes
    safe = max(1, int(TOTAL_RAM_GB / per_worker))
    return min(safe, cap)


# ── Helpers ───────────────────────────────────────────────────────────────────
def _find_best_params(gs_df):
    """Find best (lr, bs) from grid search results based on mean AUC."""
    grouped = gs_df.groupby(['batch_size', 'learning_rate'])['auc'].mean()
    best_idx = grouped.idxmax()
    best_bs, best_lr = best_idx
    best_auc = grouped[best_idx]
    print(f"  Best combo: LR={best_lr} BS={best_bs} (mean AUC={best_auc:.4f})")
    return best_lr, int(best_bs)


def _get_gs_iterations(dataset, config_name, boundary):
    """Get deterministic grid search iterations for a boundary."""
    rng = random.Random(
        SEED + boundary + hash(config_name) + hash(dataset)
    )
    return sorted(rng.sample(range(NUM_ITERATIONS), GRID_SEARCH_ITERS))


def _save_results(existing_df, new_rows, results_file, periodic=False):
    """Save results to CSV (sorted, deduplicated)."""
    new_df = pd.DataFrame(new_rows)
    all_df = pd.concat([existing_df, new_df], ignore_index=True)
    all_df = all_df.sort_values(
        by=['configuration', 'num_nodes', 'site_id', 'iteration']
    ).reset_index(drop=True)
    all_df.to_csv(results_file, index=False)
    if periodic:
        print(f"  [PERIODIC SAVE] {len(new_rows)} rows → {results_file}")


# ── Main orchestrator ─────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Fully parallel adaptive swarm learning"
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
        help=f"Max parallel workers cap (default: {PARALLEL_WORKERS}, "
             f"env: NVFLARE_SWARM_WORKERS). Actual workers are dynamically "
             f"scaled down for heavier simulations."
    )
    args = parser.parse_args()

    datasets = DATASETS if args.dataset == 'both' else [args.dataset]
    configs = CONFIGS if args.config == 'all' else {args.config: CONFIGS[args.config]}
    workers = args.workers

    print(f"\n{'#'*60}")
    print(f"ADAPTIVE SWARM LEARNING (memory-safe) - Started {datetime.now()}")
    print(f"Datasets: {datasets}")
    print(f"Configurations: {list(configs.keys())}")
    print(f"Max workers cap: {workers} (dynamic scaling by num_nodes)")
    print(f"Template: {PARALLEL_TEMPLATE}")
    print(f"Grid search: {len(BATCH_SIZES)} BS x {len(LRS)} LR "
          f"x {GRID_SEARCH_ITERS} iters per boundary")
    print(f"Group size: {GROUP_SIZE}")
    print(f"RAM budget: ~{TOTAL_RAM_GB} GB "
          f"(base {BASE_RAM_PER_WORKER_GB} + {RAM_PER_NODE_GB}/node per worker)")
    print(f"maxtasksperchild=1 + gc.collect() between batches")
    print(f"{'#'*60}")

    # Sync code ONCE before any parallel execution
    print("\nSyncing code to swarm_learning_tf/code/...")
    run_command(
        "rsync -av --exclude='dataset' ../mimic ./code/",
        cwd=SWARM_TF_DIR
    )
    print("Code sync complete.")

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

        # ── PHASE 1: Grid Searches (sequential per boundary, parallel within) ─
        print(f"\n{'='*60}")
        print(f"PHASE 1: Swarm Grid Searches (sequential per boundary)")
        print(f"{'='*60}")

        gs_loaded = {}   # (config, boundary) -> already-done DataFrame

        for config_name, max_nodes in configs.items():
            config_dir = os.path.join(dataset_dir, "same_size", config_name)
            if not os.path.isdir(config_dir):
                print(f"  WARNING: Config dir not found: {config_dir}")
                continue

            for group_end in range(GROUP_SIZE, max_nodes + 1, GROUP_SIZE):
                gs_dir = os.path.join(
                    SCRIPT_DIR, "datasets", dataset,
                    "same_size_results", config_name
                )
                os.makedirs(gs_dir, exist_ok=True)
                gs_file = os.path.join(
                    gs_dir, f"grid_search_n{group_end}.csv"
                )

                # Skip if already done
                if os.path.exists(gs_file):
                    gs_df = pd.read_csv(gs_file)
                    if len(gs_df) > 0:
                        gs_loaded[(config_name, group_end)] = gs_df
                        print(f"  [LOADED] Grid search: {config_name} "
                              f"n={group_end}")
                        continue

                # Build tasks for THIS boundary only
                gs_packed = []
                iterations = _get_gs_iterations(
                    dataset, config_name, group_end
                )
                for bs in BATCH_SIZES:
                    for lr in LRS:
                        for iteration in iterations:
                            meta = (config_name, group_end, iteration, bs, lr)
                            exp_args = (
                                dataset, config_name, group_end,
                                iteration, lr, bs
                            )
                            gs_packed.append((meta, exp_args))

                # Dynamic worker count based on num_nodes
                dyn_workers = _max_workers_for(group_end, workers)
                print(f"\n  Grid search: {config_name} n={group_end} "
                      f"({len(gs_packed)} experiments, "
                      f"{dyn_workers} workers)")

                gs_results = []
                completed = 0
                ctx = mp.get_context("spawn")
                with ctx.Pool(
                    processes=dyn_workers, initializer=_init_worker,
                    maxtasksperchild=1
                ) as pool:
                    for meta, site_results in pool.imap_unordered(
                        _gs_worker, gs_packed
                    ):
                        completed += 1
                        _, _, iteration, bs, lr = meta
                        for site_id, m in site_results:
                            gs_results.append({
                                'num_nodes': group_end,
                                'iteration': iteration,
                                'site_id': site_id,
                                'batch_size': bs,
                                'learning_rate': lr,
                                'loss': m['loss'],
                                'auc': m['auc'],
                                'auprc': m['auprc'],
                                'accuracy': m['accuracy'],
                                'precision': m['precision'],
                                'recall': m['recall'],
                            })
                        if completed % 10 == 0:
                            print(f"    [{completed}/{len(gs_packed)}] done")

                # Save immediately after each boundary
                if gs_results:
                    gs_df = pd.DataFrame(gs_results)
                    gs_df.to_csv(gs_file, index=False)
                    gs_loaded[(config_name, group_end)] = gs_df
                    print(f"  [SAVED] Grid search: {config_name} "
                          f"n={group_end} ({len(gs_results)} results)")
                else:
                    print(f"  [WARN] No results for {config_name} "
                          f"n={group_end}")

                # Free memory between boundaries
                del gs_packed, gs_results
                gc.collect()

        # Find best params for each group
        best_params = {}
        for (config_name, boundary), gs_df in gs_loaded.items():
            best_lr, best_bs = _find_best_params(gs_df)
            best_params[(config_name, boundary)] = (best_lr, best_bs)
            print(f"  BEST for {config_name} n={boundary}: "
                  f"LR={best_lr} BS={best_bs}")

        # ── PHASE 2: Swarm Runs (batched by num_nodes, adaptive workers) ──
        print(f"\n{'='*60}")
        print(f"PHASE 2: Swarm Runs (batched by num_nodes, adaptive workers)")
        print(f"{'='*60}")

        results_dir = os.path.join(SCRIPT_DIR, "results", dataset)
        os.makedirs(results_dir, exist_ok=True)
        results_file = os.path.join(results_dir, "swarm_results.csv")

        if os.path.exists(results_file):
            existing_df = pd.read_csv(results_file)
        else:
            existing_df = pd.DataFrame(columns=[
                'datetime', 'configuration', 'num_nodes', 'site_id',
                'iteration', 'lr', 'batch_size',
                'loss', 'auc', 'auprc', 'accuracy',
                'precision', 'recall'
            ])

        # Build run tasks grouped by num_nodes for batching
        tasks_by_nodes = {}  # num_nodes -> list of (meta, exp_args)

        for config_name, max_nodes in configs.items():
            config_dir = os.path.join(dataset_dir, "same_size", config_name)
            if not os.path.isdir(config_dir):
                continue

            for group_end in range(GROUP_SIZE, max_nodes + 1, GROUP_SIZE):
                group_start = max(2, group_end - GROUP_SIZE + 1)

                if (config_name, group_end) not in best_params:
                    print(f"  WARNING: No best params for "
                          f"{config_name} n={group_end}")
                    continue

                lr, bs = best_params[(config_name, group_end)]

                for num_nodes in range(group_start, group_end + 1):
                    for iteration in range(NUM_ITERATIONS):
                        mask = (
                            (existing_df['configuration'] == config_name)
                            & (existing_df['num_nodes'] == num_nodes)
                            & (existing_df['iteration'] == iteration)
                        )
                        if mask.any():
                            continue

                        meta = (config_name, num_nodes, iteration, lr, bs)
                        exp_args = (
                            dataset, config_name, num_nodes,
                            iteration, lr, bs
                        )
                        if num_nodes not in tasks_by_nodes:
                            tasks_by_nodes[num_nodes] = []
                        tasks_by_nodes[num_nodes].append((meta, exp_args))

        total_tasks = sum(len(v) for v in tasks_by_nodes.values())
        if tasks_by_nodes:
            print(f"\n  Total experiments: {total_tasks} across "
                  f"{len(tasks_by_nodes)} distinct num_nodes values")

            new_rows = []
            global_completed = 0

            # Process batches from smallest to largest num_nodes
            for num_nodes in sorted(tasks_by_nodes.keys()):
                batch = tasks_by_nodes[num_nodes]
                dyn_workers = _max_workers_for(num_nodes, workers)
                print(f"\n  Batch n={num_nodes}: {len(batch)} experiments, "
                      f"{dyn_workers} workers")

                completed = 0
                ctx = mp.get_context("spawn")
                with ctx.Pool(
                    processes=dyn_workers, initializer=_init_worker,
                    maxtasksperchild=1
                ) as pool:
                    for meta, site_results in pool.imap_unordered(
                        _run_worker, batch
                    ):
                        completed += 1
                        global_completed += 1
                        config, nn, it, lr, bs = meta
                        for site_id, m in site_results:
                            new_rows.append({
                                'datetime': datetime.now().strftime(
                                    '%Y-%m-%d %H:%M:%S'
                                ),
                                'configuration': config,
                                'num_nodes': nn,
                                'site_id': site_id,
                                'iteration': it,
                                'lr': lr,
                                'batch_size': bs,
                                'loss': m['loss'],
                                'auc': m['auc'],
                                'auprc': m['auprc'],
                                'accuracy': m['accuracy'],
                                'precision': m['precision'],
                                'recall': m['recall'],
                            })
                        if completed % 10 == 0:
                            print(f"    [{completed}/{len(batch)}] "
                                  f"(total: {global_completed}/{total_tasks})")

                # Save after each num_nodes batch completes
                if new_rows:
                    _save_results(
                        existing_df, new_rows,
                        results_file, periodic=True
                    )

                # Free memory between batches
                gc.collect()

            if new_rows:
                _save_results(
                    existing_df, new_rows, results_file, periodic=False
                )
                print(f"\n  Results saved: {results_file} "
                      f"({len(new_rows)} new rows)")
            else:
                print("  No site results collected.")
        else:
            print("  No new experiments to run.")

    print(f"\n{'#'*60}")
    print(f"ALL EXPERIMENTS COMPLETED - {datetime.now()}")
    print(f"{'#'*60}")


if __name__ == "__main__":
    main()
