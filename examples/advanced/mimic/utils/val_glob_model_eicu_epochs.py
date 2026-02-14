#!/usr/bin/env python3
import os
import json
import argparse
import sys
import pandas as pd
import numpy as np
from datetime import datetime
import random
import tensorflow as tf
import time

# Set seeds for reproducibility
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
os.environ['PYTHONHASHSEED'] = str(42)

# Add networks directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../networks'))
from mimic_nets import FCN, get_metrics

def validate_global_models(
    data_path_template,
    models_path,
    iteration,
    num_clients,
    weights,
    results_dir,
    learning_rate=0,
    total_epochs=0,
    poll_interval_sec=60,
):
    """
    Validate global models from NVFlare simulation results.
    
    Args:
        data_path_template: Template path to test data CSV file with placeholders for hospital IDs
        models_path: Path to NVFlare simulation directory
        iteration: Iteration number
        num_clients: Number of clients/nodes
        weights: List of weights for each client
        results_dir: Directory to save results
    """
    # Hospital IDs for each site
    hospital_ids = [1, 2, 3, 4, 5]

    # Reverse weights if needed (as in original code)
    weights = weights[::-1]

    # Create results directory if it doesn't exist
    os.makedirs(results_dir, exist_ok=True)
    results_file = os.path.join(results_dir, 'swarm_results.csv')

    # Expected columns including epoch for per-epoch tracking
    results_columns = [
        'datetime',
        'user',
        'splits',
        'loss',
        'auc',
        'auprc',
        'accuracy',
        'precision',
        'recall',
        'iteration',
        'epoch',
    ]

    # Track already-processed (epoch, user) from existing results for idempotency
    processed = set()

    if total_epochs <= 0:
        print("No epochs to validate (total_epochs <= 0). Exiting without evaluation.")
        return

    # Helper to append a single row to CSV (create header if new)
    def append_row(row_dict):
        row_df = pd.DataFrame([row_dict])[results_columns]
        if not os.path.exists(results_file):
            row_df.to_csv(results_file, index=False)
        else:
            row_df.to_csv(results_file, mode='a', header=False, index=False)

    # Cache datasets per client to avoid repeated loads
    cached_data = {}

    total_needed = total_epochs * num_clients
    print(f"Waiting to validate {total_needed} models across epochs and clients...")
    while len(processed) < total_needed:
        processed_in_cycle = 0
        for epoch_idx in range(1, total_epochs + 1):
            for j in range(1, num_clients + 1):
                key = (epoch_idx, j)
                if key in processed:
                    continue

                # Load test data for client j if not cached
                if j not in cached_data:
                    hospital_id = hospital_ids[j - 1]
                    data_path = data_path_template.format(hospitalid=hospital_id)
                    try:
                        data = pd.read_csv(data_path)
                        X_test = data.iloc[:, :-1].values
                        y_test = data.iloc[:, -1].values
                        cached_data[j] = (X_test, y_test, X_test.shape[1])
                    except Exception as e:
                        print(f"Error loading test data from {data_path}: {str(e)}")
                        continue

                X_test, y_test, input_dim = cached_data[j]

                # Epoch-specific model path
                model_path = os.path.join(
                    models_path,
                    f'site-{j}',
                    'simulate_job',
                    f'app_site-{j}',
                    f'site-{j}_epoch_{epoch_idx}.weights.h5'
                )

                if not os.path.exists(model_path):
                    continue

                try:
                    model = FCN(input_dim=input_dim)
                    model.build((None, input_dim))
                    model.load_weights(model_path)
                    model.compile(
                        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                        loss='binary_crossentropy',
                        metrics=get_metrics()
                    )

                    metrics = model.evaluate(X_test, y_test, verbose=0, return_dict=True)

                    weight = weights[j - 1] if weights and len(weights) >= j else (100.0 / num_clients)

                    new_row = {
                        'datetime': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'user': j,
                        'splits': weight,
                        'loss': metrics['loss'],
                        'auc': metrics['auc'],
                        'auprc': metrics.get('auprc', np.nan),
                        'accuracy': metrics['accuracy'],
                        'precision': metrics['precision'],
                        'recall': metrics['recall'],
                        'iteration': iteration,
                        'epoch': epoch_idx,
                    }

                    append_row(new_row)
                    processed.add(key)
                    processed_in_cycle += 1
                    print(f"Client {j} epoch {epoch_idx}: AUC={metrics['auc']:.4f}, Loss={metrics['loss']:.4f}")

                    # Remove the weights file after successful evaluation to free disk space
                    try:
                        os.remove(model_path)
                        print(f"Deleted weights file: {model_path}")
                    except Exception as del_err:
                        print(f"Warning: could not delete {model_path}: {del_err}")

                except Exception as e:
                    print(f"Error evaluating model at {model_path}: {str(e)}")
                    import traceback
                    traceback.print_exc()

        if processed_in_cycle == 0:
            print(f"No new models found. Sleeping {poll_interval_sec}s...")
            time.sleep(poll_interval_sec)

    print(f"\nAll evaluations completed. Results saved to {results_file}")

    # Summary statistics
    try:
        results_df = pd.read_csv(results_file)
        if not results_df.empty:
            print(f"\nSummary statistics for iteration {iteration} (all epochs):")
            print(f"  Average AUC: {results_df['auc'].mean():.4f}")
            print(f"  Average Loss: {results_df['loss'].mean():.4f}")
            print(f"  Number of validated models: {len(results_df)}")
    except Exception:
        pass

def main():
    """
    Main function that reads JSON configuration and validates models.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Validate global models from NVFlare simulation using JSON configuration."
    )
    parser.add_argument(
        'json_file', 
        type=str, 
        help="Path to JSON configuration file"
    )
    args = parser.parse_args()

    # Load JSON configuration
    try:
        with open(os.path.join("/home/swarm-learning/repos/NVFlare/examples/advanced/swarm_slost/", args.json_file), 'r') as f:
            config = json.load(f)
        print(f"Loaded configuration from: {args.json_file}")
    except FileNotFoundError:
        print(f"Error: JSON file not found: {args.json_file}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON format in {args.json_file}: {e}")
        sys.exit(1)

    # Extract parameters from JSON
    iteration = config.get("iteration", 0)
    num_clients = config.get("num_nodes", 0)
    learning_rate = config.get("hyperparameters", {}).get("learning_rate", 0)
    num_aggregation_rounds = config.get("num_aggregation_rounds", 0)
    aggregation_per_epoch = config.get("aggregation_per_epoch", 1)
    total_epochs = num_aggregation_rounds * aggregation_per_epoch

    # Extract weights from node_weights dictionary
    node_weights = config.get("node_weights", {})
    weights = []
    for i in range(1, num_clients + 1):
        weight_key = str(i)
        if weight_key in node_weights:
            weights.append(node_weights[weight_key])
        else:
            # Default to equal weight if not specified
            weights.append(100.0 / num_clients)

    # Construct paths as specified
    results_directory = config.get("results_directory", "")
    data_directory = config.get("data_directory", "")
    base_path = os.path.join("/home/swarm-learning/repos/NVFlare/examples/advanced/swarm_eicu/")

    # Template for test file paths
    test_file_template = os.path.join(base_path, data_directory, "{hospitalid}_test.csv")

    # Results directory: base_path + results_directory
    final_results_dir = os.path.join(base_path, results_directory)

    # Models path: /tmp/nvflare/mimic_swarm_{num_clients}
    models_path = f"/tmp/nvflare/mimic_swarm_{num_clients}"

    print(f"\nConfiguration:")
    print(f"  Experiment: {config.get('experiment_name', 'N/A')}")
    print(f"  Iteration: {iteration}")
    print(f"  Number of clients: {num_clients}")
    print(f"  Total epochs: {total_epochs} (num_aggregation_rounds * aggregation_per_epoch)")
    print(f"  Weights: {weights}")
    print(f"  Test file template: {test_file_template}")
    print(f"  Results directory: {final_results_dir}")
    print(f"  Models path: {models_path}")

    # Wait for models directory to appear if not present yet
    while not os.path.exists(models_path):
        print(f"\nModels directory not found at {models_path}. Waiting 60s...")
        time.sleep(60)

    # Validate models
    print(f"\nStarting model validation...")
    validate_global_models(
        data_path_template=test_file_template,
        models_path=models_path,
        iteration=iteration,
        num_clients=num_clients,
        weights=weights,
        results_dir=final_results_dir,
        learning_rate=learning_rate,
        total_epochs=total_epochs,
    )

if __name__ == "__main__":
    main()
