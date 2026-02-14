import os
import json

def create_config_files(base_dir, learning_rates, batch_sizes, iterations=5):
    # Base configuration template
    base_config = {
        "experiment_name": "",
        "num_nodes": 5,
        "node_weights": {
            "1": 20.0,
            "2": 20.0,
            "3": 20.0,
            "4": 20.0,
            "5": 20.0
        },
        "min_responses_for_aggregation": 5,
        "num_aggregation_rounds": 5,
        "aggregation_per_epoch": 5,
        "network_file": "src/nets/net_basic.py",
        "hyperparameters": {
            "learning_rate": 0.0,
            "batch_size": 0
        },
        "data_directory": f"datasets/eicu/data/",
        "results_directory": "",
        "iteration": 0
    }

    # Ensure the base directory exists
    os.makedirs(base_dir, exist_ok=True)

    # Generate files for each combination of LR, BS, and iteration
    for lr in learning_rates:
        for bs in batch_sizes:
            for iteration in range(iterations):
                # Update the configuration
                config = base_config.copy()
                config["experiment_name"] = f"eicu_5_{iteration}_lr{lr:.5f}_bs{bs}"
                config["hyperparameters"]["learning_rate"] = lr
                config["hyperparameters"]["batch_size"] = bs
                config["results_directory"] = f"results/heatmap_experiments_5nodes/5_0_lr{lr:.5f}_bs{bs}/"
                config["iteration"] = iteration

                # File name
                file_name = f"5_{iteration}_lr{lr:.5f}_bs{bs}.json".replace("0.", "0-")
                file_path = os.path.join(base_dir, file_name)

                # Write to file
                with open(file_path, "w") as f:
                    json.dump(config, f, indent=4)
                print(f"Created: {file_path}")

if __name__ == "__main__":
    base_directory = "configs"
    learning_rates = [0.0001, 0.00025, 0.0005, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.075, 0.1]
    batch_sizes = [8, 16, 32, 64, 128, 256, 512]
    create_config_files(base_directory, learning_rates, batch_sizes, 5)
