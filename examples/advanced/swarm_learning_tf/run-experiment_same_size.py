#!/usr/bin/env python3
"""
Swarm learning experiment using NVFlare simulator for same-size configurations.

Accepts command-line arguments for num_nodes, configuration, iteration, lr, bs.
Copies the correct node CSV files to /tmp/mimic_data/, edits the NVFlare config,
runs the simulator, evaluates each site's global model on the test set,
and appends metrics to a CSV results file.

Usage:
    python run-experiment_same_size.py <num_nodes> <configuration> <iteration> <lr> <bs>

Example:
    python run-experiment_same_size.py 5 10nodes 3 0.0025 512
"""

import os
import sys
import argparse
import re
import shutil
import subprocess
import pandas as pd
import numpy as np
import random
from datetime import datetime

import tensorflow as tf

sys.path.append(os.path.join(os.path.dirname(__file__), '../mimic/networks'))
from mimic_nets import FCN

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['PYTHONHASHSEED'] = str(SEED)
tf.config.experimental.enable_op_determinism()


def run_command(command, cwd=None, shell=True):
    """Execute a shell command with the swarm_env activated."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if cwd is None:
        cwd = script_dir
    activate_env = "source swarm_env/bin/activate && "
    full_command = activate_env + command
    print(f"\nExecuting: {full_command}")
    result = subprocess.run(
        full_command, shell=shell, capture_output=True, text=True,
        cwd=cwd, executable='/bin/bash'
    )
    if result.returncode != 0:
        print(f"Error executing command: {command}")
        print(f"STDOUT: {result.stdout}")
        print(f"STDERR: {result.stderr}")
    return result


def edit_config_file(config_path, min_responses_required, learning_rate, batch_size):
    """Edit the NVFlare config file with the given hyperparameters."""
    print(f"\nEditing config file: {config_path}")
    try:
        with open(config_path, 'r') as f:
            content = f.read()

        # Update min_responses_required
        content = re.sub(
            r'min_responses_required\s*=\s*\d+',
            f'min_responses_required = {min_responses_required}',
            content
        )

        # Update lr and batch_size inside mimic-learner args
        learner_pattern = r'(id = "mimic-learner"[^{]*args \{)([^}]+)(\})'

        def update_learner_args(match):
            args_section = match.group(2)
            args_section = re.sub(r'lr\s*=\s*[\d.eE+-]+', f'lr = {learning_rate}', args_section)
            args_section = re.sub(r'batch_size\s*=\s*\d+', f'batch_size = {batch_size}', args_section)
            return match.group(1) + args_section + match.group(3)

        content = re.sub(learner_pattern, update_learner_args, content, flags=re.DOTALL)

        with open(config_path, 'w') as f:
            f.write(content)
        print("Config file updated successfully")
    except Exception as e:
        print(f"Error editing config file: {e}")
        sys.exit(1)


def copy_data_files(data_dir, configuration, num_nodes, iteration):
    """Copy node CSV files to /tmp/mimic_data/ as site-{i}.csv."""
    dest_dir = "/tmp/mimic_data"
    os.makedirs(dest_dir, exist_ok=True)

    # Clean old files
    for f in os.listdir(dest_dir):
        os.remove(os.path.join(dest_dir, f))

    for i in range(1, num_nodes + 1):
        src = os.path.join(data_dir, f"node{i}_{iteration}.csv")
        dst = os.path.join(dest_dir, f"site-{i}.csv")
        if os.path.isfile(src):
            shutil.copy(src, dst)
            print(f"Copied {src} -> {dst}")
        else:
            print(f"WARNING: {src} does not exist!")


def get_metrics_fns():
    return [
        tf.keras.metrics.AUC(name='auc', curve='ROC', num_thresholds=1000),
        tf.keras.metrics.AUC(name='auprc', curve='PR', num_thresholds=1000),
        tf.keras.metrics.BinaryAccuracy(name='accuracy'),
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall'),
    ]


def evaluate_swarm_models(test_path, num_nodes, configuration, iteration, lr, bs):
    """Evaluate swarm-trained global models from NVFlare simulator output."""
    # Results file in swarm_same_size/results/
    swarm_same_size_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "swarm_same_size"
    )
    results_dir = os.path.join(swarm_same_size_dir, "results", "mimic_iii")
    os.makedirs(results_dir, exist_ok=True)
    results_file = os.path.join(results_dir, "swarm_results.csv")

    # Load test data
    test_df = pd.read_csv(test_path)
    X_test = test_df.iloc[:, :-1].values.astype(np.float32)
    y_test = test_df.iloc[:, -1].values.astype(np.float32)
    input_dim = X_test.shape[1]

    # Load or create results dataframe
    if os.path.exists(results_file):
        results_df = pd.read_csv(results_file)
    else:
        results_df = pd.DataFrame(columns=[
            'datetime', 'configuration', 'num_nodes', 'site_id',
            'iteration', 'lr', 'batch_size',
            'loss', 'auc', 'auprc', 'accuracy', 'precision', 'recall'
        ])

    job_name = f"mimic_swarm_{num_nodes}"
    models_path = f"/tmp/nvflare/{job_name}"

    for j in range(1, num_nodes + 1):
        model_path = os.path.join(models_path, f'site-{j}', 'simulate_job', f'app_site-{j}', f'site-{j}.weights.h5')

        if not os.path.exists(model_path):
            print(f"Model not found: {model_path}")
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

            new_row = {
                'datetime': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'configuration': configuration,
                'num_nodes': num_nodes,
                'site_id': j,
                'iteration': iteration,
                'lr': lr,
                'batch_size': bs,
                'loss': metrics['loss'],
                'auc': metrics['auc'],
                'auprc': metrics['auprc'],
                'accuracy': metrics['accuracy'],
                'precision': metrics['precision'],
                'recall': metrics['recall'],
            }
            results_df = pd.concat([results_df, pd.DataFrame([new_row])], ignore_index=True)
            print(f"  [Site-{j}] AUC={metrics['auc']:.4f}  AUPRC={metrics['auprc']:.4f}")
        except Exception as e:
            print(f"Error evaluating site-{j}: {e}")

    results_df.to_csv(results_file, index=False)
    print(f"Swarm results saved to {results_file}")


def main():
    parser = argparse.ArgumentParser(description="Swarm learning experiment (same-size)")
    parser.add_argument('num_nodes', type=int, help="Number of nodes/clients")
    parser.add_argument('configuration', type=str, help="Configuration folder (e.g. 10nodes)")
    parser.add_argument('iteration', type=int, help="Iteration index (0-9)")
    parser.add_argument('lr', type=float, help="Learning rate")
    parser.add_argument('bs', type=int, help="Batch size")
    args = parser.parse_args()

    num_nodes = args.num_nodes
    configuration = args.configuration
    iteration = args.iteration
    lr = args.lr
    bs = args.bs

    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Data directory (relative to swarm_same_size)
    swarm_same_size_dir = os.path.join(script_dir, "..", "swarm_same_size")
    data_dir = os.path.join(swarm_same_size_dir, "datasets", "mimic_iii", "same_size", configuration)
    test_path = os.path.join(swarm_same_size_dir, "datasets", "mimic_iii", "test.csv")

    print(f"\n{'='*60}")
    print(f"Swarm Experiment: Config={configuration} Nodes={num_nodes} Iter={iteration} LR={lr} BS={bs}")
    print(f"{'='*60}")
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Starting")

    # 1. Copy data files to /tmp/mimic_data/
    copy_data_files(data_dir, configuration, num_nodes, iteration)

    # 2. Edit NVFlare config file
    config_file_path = os.path.join(
        script_dir, "..", "..", "..", "job_templates",
        "swarm_cse_tf_model_learner", "config_fed_client.conf"
    )
    config_file_path = os.path.normpath(config_file_path)
    edit_config_file(config_file_path, num_nodes, lr, bs)

    # 3. Sync code
    print("\nSyncing code...")
    run_command("rsync -av --exclude='dataset' ../mimic ./code/")

    # 4. Create NVFlare job
    job_name = f"mimic_swarm_{num_nodes}"
    print(f"\nCreating NVFlare job: {job_name}")
    run_command(f"nvflare job create -j ./jobs/{job_name} -w swarm_cse_tf_model_learner -sd ./code -force")

    # 5. Run NVFlare simulator
    print(f"\nRunning NVFlare simulator with {num_nodes} clients")
    run_command(f"nvflare simulator ./jobs/{job_name} -w /tmp/nvflare/{job_name} -n {num_nodes} -t {num_nodes}")

    # 6. Evaluate global models and save metrics
    print("\nEvaluating swarm models...")
    evaluate_swarm_models(test_path, num_nodes, configuration, iteration, lr, bs)

    # 7. Clean up
    tmp_dir = f"/tmp/nvflare/{job_name}"
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
        print(f"Cleaned up {tmp_dir}")

    print(f"\n{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Completed")


if __name__ == "__main__":
    main()
