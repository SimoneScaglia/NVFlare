#!/usr/bin/env python3
import json
import os
import sys
import subprocess
from datetime import datetime
import re
import pandas as pd
import numpy as np
import tensorflow as tf
import time

sys.path.append(os.path.join(os.path.dirname(__file__), "../mimic/networks"))
from mimic_nets import FCN, get_metrics

def run_command(command, shell=True):
    """Execute a shell command and return the result."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    activate_env = "source swarm_env/bin/activate && "
    full_command = activate_env + command
    print(f"\nExecuting: {full_command}")
    result = subprocess.run(full_command, shell=shell, capture_output=True, text=True, cwd=script_dir, executable='/bin/bash')
    if result.returncode != 0:
        print(f"Error executing command: {command}")
        print(f"STDOUT: {result.stdout}")
        print(f"STDERR: {result.stderr}")
    return result

def edit_config_files(
    client_config_path,
    server_config_path,
    min_responses_required,
    learning_rate,
    batch_size,
    aggregation_epochs,
    num_rounds,
):
    """Edit client/server job templates with values from JSON."""
    print(f"\nEditing client config file: {client_config_path}")

    try:
        with open(client_config_path, 'r') as f:
            client_content = f.read()

        # Update min_responses_required.
        client_content = re.sub(
            r'min_responses_required\s*=\s*\d+',
            f'min_responses_required = {min_responses_required}',
            client_content,
        )

        # Use the learner variant that saves round checkpoints.
        # Replace only the learner component path, not any executor path.
        client_content = client_content.replace(
            'path = "mimic.learners.mimic_model_learner.MimicModelLearner"',
            'path = "mimic.learners.mimic_model_learner_save_weights_aggr_round.MimicModelLearner"',
        )

        # Update learner args: aggregation_epochs, lr, batch_size.
        learner_pattern = r'(id = "mimic-learner"[^{]*args \{)([^}]+)(\})'

        def update_learner_args(match):
            args_section = match.group(2)
            args_section = re.sub(r'aggregation_epochs\s*=\s*\d+', f'aggregation_epochs = {aggregation_epochs}', args_section)
            args_section = re.sub(r'lr\s*=\s*\d+\.?\d*', f'lr = {learning_rate}', args_section)
            args_section = re.sub(r'batch_size\s*=\s*\d+', f'batch_size = {batch_size}', args_section)
            return match.group(1) + args_section + match.group(3)

        client_content = re.sub(learner_pattern, update_learner_args, client_content, flags=re.DOTALL)

        with open(client_config_path, 'w') as f:
            f.write(client_content)

        print("Client config updated successfully")

        print(f"Editing server config file: {server_config_path}")
        with open(server_config_path, 'r') as f:
            server_content = f.read()

        server_content = re.sub(r'num_rounds\s*=\s*\d+', f'num_rounds = {num_rounds}', server_content)

        with open(server_config_path, 'w') as f:
            f.write(server_content)

        print("Server config updated successfully")

    except Exception as e:
        print(f"Error editing config files: {e}")
        sys.exit(1)

def copy_data_files(data_dir, iteration):
    """Copy node CSV files to /tmp/mimic_data."""
    print(f"\nCopying data files to /tmp/mimic_data/")
    ids = [1, 2, 3, 4, 5]
    
    # Create destination directory if it doesn't exist
    dest_dir = "/tmp/mimic_data"
    os.makedirs(dest_dir, exist_ok=True)
    
    for i in range(len(ids)):
        src_file = os.path.join(data_dir, f"{ids[i]}_train.csv")

        # Before copying, shuffle dataset based on iteration
        df = pd.read_csv(src_file)
        df = df.sample(frac=1, random_state=iteration).reset_index(drop=True)

        dest_file = os.path.join(dest_dir, f"site-{i + 1}.csv")
        df.to_csv(dest_file, index=False)


def append_swarm_result(path, row):
    columns = [
        "datetime",
        "user",
        "splits",
        "loss",
        "auc",
        "auprc",
        "accuracy",
        "precision",
        "recall",
        "iteration",
        "epoch",
    ]
    row_df = pd.DataFrame([{k: row.get(k, None) for k in columns}], columns=columns)
    if os.path.exists(path):
        row_df.to_csv(path, mode="a", header=False, index=False)
    else:
        row_df.to_csv(path, mode="w", header=True, index=False)


def evaluate_round_checkpoints_live(config, script_dir, simulator_proc, poll_interval_sec=5):
    num_clients = int(config.get("num_nodes", 0))
    learning_rate = config.get("hyperparameters", {}).get("learning_rate", 0.0)
    num_rounds = int(config.get("num_aggregation_rounds", 0))
    aggregation_per_epoch = int(config.get("aggregation_per_epoch", 5))
    iteration = int(config.get("iteration", 0))

    node_weights = config.get("node_weights", {})
    # Keep the historical weight ordering used by previous validation scripts.
    weights = [node_weights.get(str(i), (100.0 / num_clients if num_clients else 0.0)) for i in range(1, num_clients + 1)][::-1]

    swarm_eicu_dir = os.path.abspath(os.path.join(script_dir, "../swarm_eicu"))
    test_csv = os.path.join(swarm_eicu_dir, config.get("data_directory", ""), "test.csv")
    results_dir = os.path.join(swarm_eicu_dir, config.get("results_directory", ""))
    os.makedirs(results_dir, exist_ok=True)
    results_file = os.path.join(results_dir, "swarm_results_entire_testset.csv")

    if not os.path.exists(test_csv):
        raise FileNotFoundError(f"Test CSV not found: {test_csv}")

    data = pd.read_csv(test_csv)
    x_test = data.iloc[:, :-1].astype(np.float32).values
    y_test = data.iloc[:, -1].astype(np.float32).values
    input_dim = x_test.shape[1]

    weights_dir = "/tmp/nvflare/results/weights"
    metrics_dir = "/tmp/nvflare/results/metrics"
    print(f"\nStreaming evaluation from: {weights_dir}")

    expected = {(round_idx, client_idx) for round_idx in range(1, num_rounds + 1) for client_idx in range(1, num_clients + 1)}
    processed = set()

    while len(processed) < len(expected):
        progressed = False
        for round_idx in range(1, num_rounds + 1):
            epoch_marker = round_idx * aggregation_per_epoch
            for client_idx in range(1, num_clients + 1):
                key = (round_idx, client_idx)
                if key in processed:
                    continue

                model_path = os.path.join(weights_dir, f"nodesite-{client_idx}_round{round_idx}.weights.h5")
                if not os.path.exists(model_path):
                    continue

                model = FCN(input_dim=input_dim)
                model.build((None, input_dim))
                model.load_weights(model_path)
                model.compile(
                    optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                    loss="binary_crossentropy",
                    metrics=get_metrics(),
                )
                metrics = model.evaluate(x_test, y_test, verbose=0, return_dict=True)

                row = {
                    "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "user": client_idx,
                    "splits": weights[client_idx - 1] if len(weights) >= client_idx else 100.0 / num_clients,
                    "loss": metrics["loss"],
                    "auc": metrics["auc"],
                    "auprc": metrics.get("auprc", np.nan),
                    "accuracy": metrics["accuracy"],
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "iteration": iteration,
                    "epoch": epoch_marker,
                }
                append_swarm_result(results_file, row)
                processed.add(key)
                progressed = True

                print(
                    f"  validated round {round_idx}/{num_rounds}, client {client_idx}: "
                    f"AUC={metrics['auc']:.4f}, Loss={metrics['loss']:.4f}, epoch={epoch_marker}"
                )

                # Remove artifacts immediately to keep disk usage bounded.
                try:
                    os.remove(model_path)
                except OSError:
                    pass
                metric_path = os.path.join(metrics_dir, f"metrics_nodesite-{client_idx}_round{round_idx}.json")
                if os.path.exists(metric_path):
                    try:
                        os.remove(metric_path)
                    except OSError:
                        pass

                tf.keras.backend.clear_session()

        if len(processed) == len(expected):
            break

        if simulator_proc.poll() is not None and not progressed:
            missing = sorted(expected - processed)
            sample = ", ".join([f"r{r}-c{c}" for r, c in missing[:10]])
            raise RuntimeError(
                f"Simulator finished but {len(missing)} checkpoints are missing. Sample missing: {sample}"
            )

        if not progressed:
            time.sleep(poll_interval_sec)

    # Ensure simulator completed successfully.
    simulator_rc = simulator_proc.wait()
    if simulator_rc != 0:
        raise RuntimeError(f"nvflare simulator exited with code {simulator_rc}")

    print(f"\nSaved per-round swarm metrics to: {results_file}")

def main():
    # Check if JSON file path is provided
    if len(sys.argv) < 2:
        print("Usage: python script.py <path_to_json_file>")
        sys.exit(1)
    
    json_path = sys.argv[1]
    
    # Print current datetime
    print(f"\n{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Starting script")
    
    # Load JSON file
    try:
        with open(json_path, 'r') as f:
            config = json.load(f)
        
        print(f"Loaded configuration from: {json_path}")
        print(f"Experiment: {config.get('experiment_name')}")
        
    except FileNotFoundError:
        print(f"Error: JSON file not found: {json_path}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON format: {e}")
        sys.exit(1)
    
    # Activate virtual environment
    print("\nActivating virtual environment...")
    run_command("source swarm_env/bin/activate", shell=True)
    
    # Edit config files
    client_config_file_path = "../../../job_templates/swarm_cse_tf_model_learner/config_fed_client.conf"
    server_config_file_path = "../../../job_templates/swarm_cse_tf_model_learner/config_fed_server.conf"
    
    # Get values from JSON
    min_responses_for_aggregation = config.get("min_responses_for_aggregation", 0)
    learning_rate = config.get("hyperparameters", {}).get("learning_rate", 0)
    batch_size = config.get("hyperparameters", {}).get("batch_size", 0)
    aggregation_per_epoch = config.get("aggregation_per_epoch", 5)
    num_aggregation_rounds = config.get("num_aggregation_rounds", 1)
    
    # Edit the config files
    edit_config_files(
        client_config_path=client_config_file_path,
        server_config_path=server_config_file_path,
        min_responses_required=min_responses_for_aggregation,
        learning_rate=learning_rate,
        batch_size=batch_size,
        aggregation_epochs=aggregation_per_epoch,
        num_rounds=num_aggregation_rounds,
    )
    
    # Copy data files
    data_directory = config.get("data_directory", "")
    iteration = config.get("iteration", 0)
    num_nodes = config.get("num_nodes", 0)
    
    copy_data_files(data_directory, iteration)
    
    # Rsync command
    print("\nSyncing code...")
    rsync_result = run_command("rsync -av --exclude='dataset' ../mimic ./code/")
    if rsync_result.returncode != 0:
        print("Warning: Rsync command failed")
    
    # Create nvflare job
    num_clients = num_nodes
    job_name = f"mimic_swarm_{num_clients}"
    print(f"\nCreating nvflare job: {job_name}")

    # Clean old checkpoint artifacts to avoid mixing runs.
    run_command("rm -rf /tmp/nvflare/results/weights /tmp/nvflare/results/metrics")
    
    create_job_cmd = f"nvflare job create -j ./jobs/{job_name} -w swarm_cse_tf_model_learner -sd ./code -force"
    create_result = run_command(create_job_cmd)
    if create_result.returncode != 0:
        sys.exit(create_result.returncode)
    
    # Run nvflare simulator
    print(f"\nRunning nvflare simulator with {num_clients} clients")
    simulator_cmd = f"source swarm_env/bin/activate && nvflare simulator ./jobs/{job_name} -w /tmp/nvflare/{job_name} -n {num_clients} -t {num_clients}"
    simulator_proc = subprocess.Popen(
        simulator_cmd,
        shell=True,
        cwd=os.path.dirname(os.path.abspath(__file__)),
        executable="/bin/bash",
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    # Evaluate and save one row per client every 5 epochs while simulator is still running.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    evaluate_round_checkpoints_live(config=config, script_dir=script_dir, simulator_proc=simulator_proc)
    
    # Clean up temporary directory
    print(f"\nCleaning up temporary directory...")
    temp_dir = f"/tmp/nvflare/mimic_swarm_{num_clients}"
    if os.path.exists(temp_dir):
        run_command(f"rm -rf \"{temp_dir}\"")
        print(f"Removed: {temp_dir}")
    else:
        print(f"Directory does not exist: {temp_dir}")
    
    # Print final timestamp
    print(f"\n{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Script completed")

if __name__ == "__main__":
    main()
