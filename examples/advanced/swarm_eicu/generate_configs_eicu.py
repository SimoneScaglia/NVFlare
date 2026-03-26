import argparse
import copy
import json
import os
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Generate experiment configs for swarm_eicu experiments.")
    parser.add_argument("--base-dir", type=str, default="configs_all_datasets_epochs", help="Output directory for JSON configs")
    parser.add_argument("--iterations", type=int, default=5, help="Number of iterations per configuration")
    return parser.parse_args()


def create_config_files(base_dir, learning_rates, batch_sizes, epoch_values, iterations=5):
    base_config = {
        "experiment_name": "",
        "num_nodes": 5,
        "node_weights": {
            "1": 20.0,
            "2": 20.0,
            "3": 20.0,
            "4": 20.0,
            "5": 20.0,
        },
        "min_responses_for_aggregation": 5,
        "num_aggregation_rounds": 1,
        "aggregation_per_epoch": 5,
        "network_file": "src/nets/net_basic.py",
        "hyperparameters": {
            "learning_rate": 0.0,
            "batch_size": 0,
        },
        "data_directory": "",
        "results_directory": "",
        "iteration": 0,
    }

    datasets = {
        # "eicu": {
        #     "data_directory": "datasets/eicu/data/",
        #     "results_subdir": "eicu_entire_testset_epochs",
        # },
        "mimiciii": {
            "data_directory": "datasets/mimiciii/",
            "results_subdir": "mimiciii_total_entire_testset_epochs",
        },
        "mimiciv": {
            "data_directory": "datasets/mimiciv/total/",
            "results_subdir": "mimiciv_total_entire_testset_epochs",
        },
        "mimiciv_fixed": {
            "data_directory": "datasets/mimiciv/fixed_12500/",
            "results_subdir": "mimiciv_fixed_12500_entire_testset_epochs",
        },
    }

    os.makedirs(base_dir, exist_ok=True)

    # Remove old generated JSONs to avoid stale runs when parameters change.
    for old_json in Path(base_dir).glob("*.json"):
        old_json.unlink()

    created = 0
    for dataset_name, dataset_cfg in datasets.items():
        for total_epochs in epoch_values:
            aggregation_per_epoch = 5
            num_rounds = total_epochs // aggregation_per_epoch

            for lr in learning_rates:
                for bs in batch_sizes:
                    for iteration in range(iterations):
                        config = copy.deepcopy(base_config)
                        config["experiment_name"] = f"{dataset_name}_{total_epochs}_{iteration}_lr{lr:.5f}_bs{bs}"
                        config["num_aggregation_rounds"] = num_rounds
                        config["aggregation_per_epoch"] = aggregation_per_epoch
                        config["hyperparameters"]["learning_rate"] = lr
                        config["hyperparameters"]["batch_size"] = bs
                        config["data_directory"] = dataset_cfg["data_directory"]
                        config["results_directory"] = (
                            f"new_results/{dataset_cfg['results_subdir']}/{total_epochs}_{iteration}_lr{lr:.5f}_bs{bs}/"
                        )
                        config["iteration"] = iteration

                        file_name = (
                            f"{dataset_name}_{total_epochs}_{iteration}_lr{lr:.5f}_bs{bs}.json".replace("0.", "0-")
                        )
                        file_path = os.path.join(base_dir, file_name)

                        with open(file_path, "w", encoding="utf-8") as f:
                            json.dump(config, f, indent=4)
                        created += 1

    print(f"Generated {created} config files in: {base_dir}")


if __name__ == "__main__":
    args = parse_args()

    learning_rates = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1]
    batch_sizes = [8, 16, 32, 64, 128, 256, 512]
    epoch_values = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]

    create_config_files(
        base_dir=args.base_dir,
        learning_rates=learning_rates,
        batch_sizes=batch_sizes,
        epoch_values=epoch_values,
        iterations=args.iterations,
    )
