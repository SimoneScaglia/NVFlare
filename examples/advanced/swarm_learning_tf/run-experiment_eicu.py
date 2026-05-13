#!/usr/bin/env python3

import gc
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

sys.path.append(os.path.join(os.path.dirname(__file__), "../mimic/networks"))
from mimic_nets import FCN, get_metrics


ID_TRAIN_RE = re.compile(r"^(\d+)_train\.csv$")

SWARM_OUTPUT_COLUMNS = [
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


def run_command(command, shell=True):
    """Execute a shell command from this script directory inside swarm_env."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    activate_env = "source swarm_env/bin/activate && "
    full_command = activate_env + command
    print(f"\nExecuting: {full_command}")
    result = subprocess.run(
        full_command,
        shell=shell,
        capture_output=True,
        text=True,
        cwd=script_dir,
        executable="/bin/bash",
    )
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
        with open(client_config_path, "r", encoding="utf-8") as f:
            client_content = f.read()

        client_content = re.sub(
            r"min_responses_required\s*=\s*\d+",
            f"min_responses_required = {min_responses_required}",
            client_content,
        )

        client_content = client_content.replace(
            'path = "mimic.learners.mimic_model_learner.MimicModelLearner"',
            'path = "mimic.learners.mimic_model_learner_save_weights_aggr_round.MimicModelLearner"',
        )

        learner_pattern = r'(id = "mimic-learner"[^{]*args \{)([^}]+)(\})'

        def update_learner_args(match):
            args_section = match.group(2)
            args_section = re.sub(r"aggregation_epochs\s*=\s*\d+", f"aggregation_epochs = {aggregation_epochs}", args_section)
            args_section = re.sub(r"lr\s*=\s*\d+\.?\d*", f"lr = {learning_rate}", args_section)
            args_section = re.sub(r"batch_size\s*=\s*\d+", f"batch_size = {batch_size}", args_section)
            return match.group(1) + args_section + match.group(3)

        client_content = re.sub(learner_pattern, update_learner_args, client_content, flags=re.DOTALL)

        with open(client_config_path, "w", encoding="utf-8") as f:
            f.write(client_content)

        print("Client config updated successfully")

        print(f"Editing server config file: {server_config_path}")
        with open(server_config_path, "r", encoding="utf-8") as f:
            server_content = f.read()

        server_content = re.sub(r"num_rounds\s*=\s*\d+", f"num_rounds = {num_rounds}", server_content)

        with open(server_config_path, "w", encoding="utf-8") as f:
            f.write(server_content)

        print("Server config updated successfully")

    except Exception as e:
        print(f"Error editing config files: {e}")
        sys.exit(1)


def discover_dataset_ids(data_dir: Path):
    ids = []
    for p in sorted(data_dir.glob("*_train.csv")):
        m = ID_TRAIN_RE.match(p.name)
        if not m:
            continue
        dataset_id = int(m.group(1))
        if (data_dir / f"{dataset_id}_test.csv").exists():
            ids.append(dataset_id)

    if not ids:
        raise ValueError(f"No <id>_train.csv / <id>_test.csv pairs found in {data_dir}")
    return ids


def copy_data_files(data_dir: Path, dataset_ids, iteration: int):
    """Copy and shuffle node train CSV files into /tmp/mimic_data as site-{i}.csv."""
    print("\nCopying train data files to /tmp/mimic_data/")

    dest_dir = Path("/tmp/mimic_data")
    dest_dir.mkdir(parents=True, exist_ok=True)

    for stale in dest_dir.glob("site-*.csv"):
        try:
            stale.unlink()
        except OSError:
            pass

    for site_index, dataset_id in enumerate(dataset_ids, start=1):
        src_file = data_dir / f"{dataset_id}_train.csv"
        if not src_file.exists():
            raise FileNotFoundError(f"Missing train file: {src_file}")

        df = pd.read_csv(src_file)
        df = df.sample(frac=1, random_state=iteration).reset_index(drop=True)

        dest_file = dest_dir / f"site-{site_index}.csv"
        df.to_csv(dest_file, index=False)


def append_swarm_result(path: Path, row: dict):
    row_df = pd.DataFrame([{k: row.get(k, None) for k in SWARM_OUTPUT_COLUMNS}], columns=SWARM_OUTPUT_COLUMNS)
    if path.exists():
        row_df.to_csv(path, mode="a", header=False, index=False)
    else:
        row_df.to_csv(path, mode="w", header=True, index=False)


def load_concat_test_df(data_dir: Path, dataset_ids, iteration: int):
    frames = []
    for dataset_id in dataset_ids:
        test_path = data_dir / f"{dataset_id}_test.csv"
        if not test_path.exists():
            raise FileNotFoundError(f"Missing test file: {test_path}")
        frames.append(pd.read_csv(test_path))

    df = pd.concat(frames, axis=0, ignore_index=True)
    return df.sample(frac=1, random_state=iteration).reset_index(drop=True)


def evaluate_with_model_weights(model_path: Path, x_test: np.ndarray, y_test: np.ndarray, input_dim: int, learning_rate: float):
    import time
    import errno
    
    model = FCN(input_dim=input_dim)
    model.build((None, input_dim))
    
    # Retry logic for file locking issues with h5py
    max_retries = 5
    retry_delay = 0.5  # seconds
    
    for attempt in range(max_retries):
        try:
            model.load_weights(str(model_path))
            break
        except (BlockingIOError, OSError) as e:
            if attempt < max_retries - 1:
                if isinstance(e, BlockingIOError) or (isinstance(e, OSError) and e.errno == errno.EWOULDBLOCK):
                    wait_time = retry_delay * (2 ** attempt)
                    print(f"  File locking issue on {model_path.name}, retrying in {wait_time:.2f}s (attempt {attempt + 1}/{max_retries})")
                    time.sleep(wait_time)
                else:
                    raise
            else:
                raise RuntimeError(f"Failed to load weights from {model_path} after {max_retries} attempts: {e}")
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=get_metrics(),
    )
    metrics = model.evaluate(x_test, y_test, verbose=0, return_dict=True)

    tf.keras.backend.clear_session()
    del model
    gc.collect()

    return metrics


def evaluate_round_checkpoints_live(
    config: dict,
    script_dir: Path,
    data_dir: Path,
    dataset_ids,
    simulator_proc,
    poll_interval_sec: int = 5,
):
    num_clients = len(dataset_ids)
    iteration = int(config.get("iteration", 0))
    learning_rate = float(config.get("hyperparameters", {}).get("learning_rate", 0.0))
    eval_mode = str(config.get("evaluation_mode", "entire")).strip().lower()
    num_rounds = int(config.get("num_aggregation_rounds", 0))
    aggregation_per_epoch = int(config.get("aggregation_per_epoch", 5))

    if num_rounds <= 0:
        raise ValueError("num_aggregation_rounds must be > 0")
    if aggregation_per_epoch <= 0:
        raise ValueError("aggregation_per_epoch must be > 0")

    node_weights = config.get("node_weights", {})
    raw_client_weights = np.array(
        [float(node_weights.get(str(i), 100.0 / num_clients)) for i in range(1, num_clients + 1)],
        dtype=np.float64,
    )
    if raw_client_weights.sum() <= 0:
        raw_client_weights = np.ones(num_clients, dtype=np.float64)
    normalized_weights = raw_client_weights / raw_client_weights.sum()

    swarm_eicu_dir = (script_dir / "../swarm_eicu").resolve()

    results_dir = Path(config.get("results_directory", ""))
    if not results_dir.is_absolute():
        results_dir = (swarm_eicu_dir / results_dir).resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    weights_dir = Path("/tmp/nvflare/results/weights")
    metrics_dir = Path("/tmp/nvflare/results/metrics")

    expected_rounds = set(range(1, num_rounds + 1))
    processed_rounds = set()

    if eval_mode == "separate":
        results_file = results_dir / "swarm_results.csv"
        default_split = 100.0 / float(num_clients)

        test_cache = {}
        for dataset_id in dataset_ids:
            test_df = pd.read_csv(data_dir / f"{dataset_id}_test.csv")
            test_cache[dataset_id] = {
                "x": test_df.iloc[:, :-1].astype(np.float32).values,
                "y": test_df.iloc[:, -1].astype(np.float32).values,
            }

        while len(processed_rounds) < len(expected_rounds):
            progressed = False
            for round_idx in range(1, num_rounds + 1):
                if round_idx in processed_rounds:
                    continue

                epoch_marker = round_idx * aggregation_per_epoch
                model_paths = [weights_dir / f"nodesite-{site_idx}_round{round_idx}.weights.h5" for site_idx in range(1, num_clients + 1)]
                if not all(path.exists() for path in model_paths):
                    continue

                for site_idx, dataset_id in enumerate(dataset_ids, start=1):
                    test_data = test_cache[dataset_id]
                    metrics = evaluate_with_model_weights(
                        model_path=model_paths[site_idx - 1],
                        x_test=test_data["x"],
                        y_test=test_data["y"],
                        input_dim=test_data["x"].shape[1],
                        learning_rate=learning_rate,
                    )

                    row = {
                        "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "user": dataset_id,
                        "splits": float(node_weights.get(str(site_idx), default_split)),
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

                for site_idx, model_path in enumerate(model_paths, start=1):
                    try:
                        model_path.unlink()
                    except OSError:
                        pass
                    metric_path = metrics_dir / f"metrics_nodesite-{site_idx}_round{round_idx}.json"
                    if metric_path.exists():
                        try:
                            metric_path.unlink()
                        except OSError:
                            pass

                processed_rounds.add(round_idx)
                progressed = True
                print(f"  validated round {round_idx}/{num_rounds} (epoch={epoch_marker}) for separate mode")

            if len(processed_rounds) == len(expected_rounds):
                break

            if simulator_proc.poll() is not None and not progressed:
                missing = sorted(expected_rounds - processed_rounds)
                sample = ", ".join([f"r{r}" for r in missing[:10]])
                raise RuntimeError(
                    f"Simulator finished but {len(missing)} rounds are missing. Sample: {sample}"
                )

            if not progressed:
                import time

                time.sleep(poll_interval_sec)

        simulator_rc = simulator_proc.wait()
        if simulator_rc != 0:
            raise RuntimeError(f"nvflare simulator exited with code {simulator_rc}")

        print(f"\nSaved swarm separate-testset per-round results to: {results_file}")
        return

    if eval_mode == "entire":
        results_file = results_dir / "swarm_results_entire_testset.csv"

        test_df = load_concat_test_df(data_dir=data_dir, dataset_ids=dataset_ids, iteration=iteration)
        x_test = test_df.iloc[:, :-1].astype(np.float32).values
        y_test = test_df.iloc[:, -1].astype(np.float32).values
        input_dim = x_test.shape[1]

        while len(processed_rounds) < len(expected_rounds):
            progressed = False
            for round_idx in range(1, num_rounds + 1):
                if round_idx in processed_rounds:
                    continue

                epoch_marker = round_idx * aggregation_per_epoch
                model_paths = [weights_dir / f"nodesite-{site_idx}_round{round_idx}.weights.h5" for site_idx in range(1, num_clients + 1)]
                if not all(path.exists() for path in model_paths):
                    continue

                local_weight_lists = []
                for model_path in model_paths:
                    model = FCN(input_dim=input_dim)
                    model.build((None, input_dim))
                    model.load_weights(str(model_path))
                    local_weight_lists.append(model.get_weights())
                    tf.keras.backend.clear_session()
                    del model
                    gc.collect()

                aggregated_weights = []
                for layer_idx in range(len(local_weight_lists[0])):
                    layer_stack = np.stack([client_layers[layer_idx] for client_layers in local_weight_lists], axis=0)
                    layer_avg = np.tensordot(normalized_weights, layer_stack, axes=(0, 0))
                    aggregated_weights.append(layer_avg)

                global_model = FCN(input_dim=input_dim)
                global_model.build((None, input_dim))
                global_model.set_weights(aggregated_weights)
                global_model.compile(
                    optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                    loss="binary_crossentropy",
                    metrics=get_metrics(),
                )

                metrics = global_model.evaluate(x_test, y_test, verbose=0, return_dict=True)

                row = {
                    "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "user": "swarm_global",
                    "splits": 100,
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

                tf.keras.backend.clear_session()
                del global_model
                gc.collect()

                for site_idx, model_path in enumerate(model_paths, start=1):
                    try:
                        model_path.unlink()
                    except OSError:
                        pass
                    metric_path = metrics_dir / f"metrics_nodesite-{site_idx}_round{round_idx}.json"
                    if metric_path.exists():
                        try:
                            metric_path.unlink()
                        except OSError:
                            pass

                processed_rounds.add(round_idx)
                progressed = True
                print(
                    f"  validated aggregated round {round_idx}/{num_rounds}: "
                    f"AUC={metrics['auc']:.4f}, Loss={metrics['loss']:.4f}, epoch={epoch_marker}"
                )

            if len(processed_rounds) == len(expected_rounds):
                break

            if simulator_proc.poll() is not None and not progressed:
                missing = sorted(expected_rounds - processed_rounds)
                sample = ", ".join([f"r{r}" for r in missing[:10]])
                raise RuntimeError(
                    f"Simulator finished but {len(missing)} rounds are missing. Sample: {sample}"
                )

            if not progressed:
                import time

                time.sleep(poll_interval_sec)

        simulator_rc = simulator_proc.wait()
        if simulator_rc != 0:
            raise RuntimeError(f"nvflare simulator exited with code {simulator_rc}")

        print(f"\nSaved swarm entire-testset per-round results to: {results_file}")
        return

    raise ValueError(f"Unsupported evaluation_mode: {eval_mode}. Expected 'separate' or 'entire'.")


def main():
    if len(sys.argv) < 2:
        print("Usage: python run-experiment_eicu.py <path_to_json_file>")
        sys.exit(1)

    json_path = Path(sys.argv[1]).resolve()

    print(f"\n{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Starting script")

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        print(f"Loaded configuration from: {json_path}")
        print(f"Experiment: {config.get('experiment_name')}")

    except FileNotFoundError:
        print(f"Error: JSON file not found: {json_path}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON format: {e}")
        sys.exit(1)

    script_dir = Path(__file__).resolve().parent
    swarm_eicu_dir = (script_dir / "../swarm_eicu").resolve()

    client_config_file_path = "../../../job_templates/swarm_cse_tf_model_learner/config_fed_client.conf"
    server_config_file_path = "../../../job_templates/swarm_cse_tf_model_learner/config_fed_server.conf"

    min_responses_for_aggregation = int(config.get("min_responses_for_aggregation", 0))
    learning_rate = float(config.get("hyperparameters", {}).get("learning_rate", 0.0))
    batch_size = int(config.get("hyperparameters", {}).get("batch_size", 0))
    aggregation_per_epoch = int(config.get("aggregation_per_epoch", 1))
    num_aggregation_rounds = int(config.get("num_aggregation_rounds", 1))

    edit_config_files(
        client_config_path=client_config_file_path,
        server_config_path=server_config_file_path,
        min_responses_required=min_responses_for_aggregation,
        learning_rate=learning_rate,
        batch_size=batch_size,
        aggregation_epochs=aggregation_per_epoch,
        num_rounds=num_aggregation_rounds,
    )

    data_directory = Path(config.get("data_directory", ""))
    if not data_directory.is_absolute():
        data_directory = (swarm_eicu_dir / data_directory).resolve()

    if not data_directory.exists():
        raise FileNotFoundError(f"Data directory not found: {data_directory}")

    dataset_ids = config.get("dataset_ids") or discover_dataset_ids(data_directory)
    dataset_ids = [int(x) for x in dataset_ids]

    num_nodes = int(config.get("num_nodes", len(dataset_ids)))
    if num_nodes != len(dataset_ids):
        raise ValueError(f"num_nodes ({num_nodes}) does not match dataset_ids count ({len(dataset_ids)})")

    iteration = int(config.get("iteration", 0))

    copy_data_files(data_dir=data_directory, dataset_ids=dataset_ids, iteration=iteration)

    print("\nSyncing code...")
    rsync_result = run_command("rsync -av --exclude='dataset' ../mimic ./code/")
    if rsync_result.returncode != 0:
        print("Warning: rsync command failed")

    job_name = f"mimic_swarm_{num_nodes}"
    print(f"\nCreating nvflare job: {job_name}")

    run_command("rm -rf /tmp/nvflare/results/weights /tmp/nvflare/results/metrics")

    create_job_cmd = f"nvflare job create -j ./jobs/{job_name} -w swarm_cse_tf_model_learner -sd ./code -force"
    create_result = run_command(create_job_cmd)
    if create_result.returncode != 0:
        sys.exit(create_result.returncode)

    print(f"\nRunning nvflare simulator with {num_nodes} clients")
    simulator_cmd = (
        "source swarm_env/bin/activate && "
        f"nvflare simulator ./jobs/{job_name} -w /tmp/nvflare/{job_name} -n {num_nodes} -t {num_nodes}"
    )
    simulator_proc = subprocess.Popen(
        simulator_cmd,
        shell=True,
        cwd=str(script_dir),
        executable="/bin/bash",
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    evaluate_round_checkpoints_live(
        config=config,
        script_dir=script_dir,
        data_dir=data_directory,
        dataset_ids=dataset_ids,
        simulator_proc=simulator_proc,
    )

    print("\nCleaning up temporary directory...")
    temp_dir = Path(f"/tmp/nvflare/mimic_swarm_{num_nodes}")
    if temp_dir.exists():
        run_command(f"rm -rf \"{temp_dir}\"")
        print(f"Removed: {temp_dir}")
    else:
        print(f"Directory does not exist: {temp_dir}")

    print(f"\n{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Script completed")


if __name__ == "__main__":
    main()
