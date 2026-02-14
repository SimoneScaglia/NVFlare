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

# Set seeds for reproducibility
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
os.environ['PYTHONHASHSEED'] = str(42)

# Add networks directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../networks'))
from mimic_nets import FCN, get_metrics

def validate_global_models(data_path_template, models_path, iteration, num_clients, weights, results_dir, learning_rate=0):
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

    # Load existing results or create new dataframe
    if os.path.exists(results_file):
        results_df = pd.read_csv(results_file)
    else:
        results_df = pd.DataFrame(columns=[
            'datetime',
            'user',
            'splits',
            'loss',
            'auc',
            'auprc',
            'accuracy',
            'precision',
            'recall',
            'iteration'
        ])

    # Validate models for each client
    for j in range(1, num_clients + 1):
        # Construct test file path for the current hospital ID
        hospital_id = hospital_ids[j - 1]
        data_path = data_path_template.format(hospitalid=hospital_id)

        # Load test data
        try:
            data = pd.read_csv(data_path)
            X_test = data.iloc[:, :-1].values
            y_test = data.iloc[:, -1].values
        except Exception as e:
            print(f"Error loading test data from {data_path}: {str(e)}")
            continue

        input_dim = X_test.shape[1]

        # Construct model path based on NVFlare directory structure
        model_path = os.path.join(
            models_path, 
            f'site-{j}', 
            'simulate_job', 
            f'app_site-{j}', 
            f'site-{j}.weights.h5'
        )

        if not os.path.exists(model_path):
            print(f"Model not found: {model_path}")
            continue

        try:
            # Create and load model
            model = FCN(input_dim=input_dim)
            model.build((None, input_dim))
            model.load_weights(model_path)
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                loss='binary_crossentropy', 
                metrics=get_metrics()
            )

            # Evaluate model
            metrics = model.evaluate(X_test, y_test, verbose=0, return_dict=True)

            # Get weight for this client (default to equal distribution if not provided)
            if weights and len(weights) >= j:
                weight = weights[j - 1]
            else:
                weight = 100.0 / num_clients

            # Create results row
            new_row = {
                'datetime': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'user': j,  # Using j directly since is_data is not defined
                'splits': weight,
                'loss': metrics['loss'],
                'auc': metrics['auc'],
                'auprc': metrics.get('auprc', np.nan),  # Use .get() for optional metrics
                'accuracy': metrics['accuracy'],
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'iteration': iteration
            }

            # Append to results
            results_df = pd.concat([results_df, pd.DataFrame([new_row])], ignore_index=True)
            print(f"Client {j}: AUC={metrics['auc']:.4f}, Loss={metrics['loss']:.4f}")

        except Exception as e:
            print(f"Error evaluating model at {model_path}: {str(e)}")
            import traceback
            traceback.print_exc()

    # Save results
    results_df.to_csv(results_file, index=False)
    print(f"\nResults saved to {results_file}")

    # Print summary statistics
    if not results_df.empty:
        print(f"\nSummary statistics for iteration {iteration}:")
        print(f"  Average AUC: {results_df['auc'].mean():.4f}")
        print(f"  Average Loss: {results_df['loss'].mean():.4f}")
        print(f"  Number of validated models: {len(results_df)}")

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
    print(f"  Weights: {weights}")
    print(f"  Test file template: {test_file_template}")
    print(f"  Results directory: {final_results_dir}")
    print(f"  Models path: {models_path}")

    # Check if models directory exists
    if not os.path.exists(models_path):
        print(f"\nError: Models directory not found at {models_path}")
        print("Make sure the NVFlare simulation has run successfully.")
        sys.exit(1)

    # Validate models
    print(f"\nStarting model validation...")
    validate_global_models(
        data_path_template=test_file_template,
        models_path=models_path,
        iteration=iteration,
        num_clients=num_clients,
        weights=weights,
        results_dir=final_results_dir,
        learning_rate=learning_rate
    )

if __name__ == "__main__":
    main()
