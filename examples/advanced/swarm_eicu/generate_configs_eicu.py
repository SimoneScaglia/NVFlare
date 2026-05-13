import argparse
import copy
import json
from pathlib import Path
import re


TRAIN_RE = re.compile(r"^(\d+)_train\.csv$")
TEST_RE = re.compile(r"^(\d+)_test\.csv$")


def parse_args():
    parser = argparse.ArgumentParser(description="Generate eICU experiment configs for swarm_eicu.")
    parser.add_argument("--base-dir", type=str, default="configs_eicu", help="Output directory for JSON configs")
    parser.add_argument("--iterations", type=int, default=5, help="Number of iterations per configuration")
    parser.add_argument("--total-epochs", type=int, default=150, help="Total epochs for each run")
    parser.add_argument(
        "--aggregation-per-epoch",
        type=int,
        default=5,
        help="Local epochs per aggregation round",
    )
    return parser.parse_args()


def discover_dataset_ids(data_dir: Path):
    train_ids = set()
    test_ids = set()

    for p in data_dir.glob("*_train.csv"):
        m = TRAIN_RE.match(p.name)
        if m:
            train_ids.add(int(m.group(1)))

    for p in data_dir.glob("*_test.csv"):
        m = TEST_RE.match(p.name)
        if m:
            test_ids.add(int(m.group(1)))

    ids = sorted(train_ids & test_ids)
    if not ids:
        raise ValueError(f"No valid <id>_train.csv / <id>_test.csv pairs found in: {data_dir}")

    return ids


def build_uniform_node_weights(num_nodes: int):
    if num_nodes <= 0:
        raise ValueError("num_nodes must be > 0")
    share = 100.0 / float(num_nodes)
    return {str(i): share for i in range(1, num_nodes + 1)}


def create_config_files(base_dir, learning_rates, batch_sizes, total_epochs, aggregation_per_epoch=5, iterations=5):
    if total_epochs <= 0:
        raise ValueError("total_epochs must be > 0")
    if aggregation_per_epoch <= 0:
        raise ValueError("aggregation_per_epoch must be > 0")
    if total_epochs % aggregation_per_epoch != 0:
        raise ValueError("total_epochs must be divisible by aggregation_per_epoch")

    base_config = {
        "experiment_name": "",
        "num_nodes": 0,
        "node_weights": {},
        "min_responses_for_aggregation": 0,
        "num_aggregation_rounds": 1,
        "aggregation_per_epoch": 5,
        "network_file": "src/nets/net_basic.py",
        "hyperparameters": {
            "learning_rate": 0.0,
            "batch_size": 0,
        },
        "data_directory": "",
        "dataset_ids": [],
        "evaluation_mode": "entire",
        "results_directory": "",
        "iteration": 0,
    }

    # The 4 eICU datasets requested by the user.
    dataset_variants = ["data", "data_fixed_rows", "data_20k", "data_20k_fixed_rows"]
    eval_modes = ["separate", "entire"]

    base_dir_path = Path(base_dir)
    base_dir_path.mkdir(parents=True, exist_ok=True)

    # Remove stale JSON files from previous generations.
    for old_json in base_dir_path.glob("*.json"):
        old_json.unlink()

    swarm_eicu_dir = Path(__file__).resolve().parent
    num_rounds = total_epochs // aggregation_per_epoch

    created = 0
    for dataset_variant in dataset_variants:
        data_dir = swarm_eicu_dir / "datasets" / "eicu" / dataset_variant
        if not data_dir.exists():
            raise FileNotFoundError(f"Dataset directory not found: {data_dir}")

        dataset_ids = discover_dataset_ids(data_dir)
        num_nodes = len(dataset_ids)
        node_weights = build_uniform_node_weights(num_nodes)

        for eval_mode in eval_modes:
            results_subdir = f"eicu_{dataset_variant}_{eval_mode}_testset"
            min_responses = num_nodes

            for lr in learning_rates:
                for bs in batch_sizes:
                    for iteration in range(iterations):
                        config = copy.deepcopy(base_config)
                        config["experiment_name"] = (
                            f"eicu_{dataset_variant}_{eval_mode}_{total_epochs}_{iteration}_lr{lr:.5f}_bs{bs}"
                        )
                        config["num_nodes"] = num_nodes
                        config["node_weights"] = node_weights
                        config["min_responses_for_aggregation"] = min_responses
                        config["num_aggregation_rounds"] = num_rounds
                        config["aggregation_per_epoch"] = aggregation_per_epoch
                        config["hyperparameters"]["learning_rate"] = lr
                        config["hyperparameters"]["batch_size"] = bs
                        config["data_directory"] = f"datasets/eicu/{dataset_variant}/"
                        config["dataset_ids"] = dataset_ids
                        config["evaluation_mode"] = eval_mode
                        config["results_directory"] = (
                            f"new_results/{results_subdir}/{total_epochs}_{iteration}_lr{lr:.5f}_bs{bs}/"
                        )
                        config["iteration"] = iteration

                        file_name = (
                            f"eicu_{dataset_variant}_{eval_mode}_{total_epochs}_{iteration}_lr{lr:.5f}_bs{bs}.json"
                        ).replace("0.", "0-")
                        file_path = base_dir_path / file_name

                        with open(file_path, "w", encoding="utf-8") as f:
                            json.dump(config, f, indent=4)
                        created += 1

    print(f"Generated {created} config files in: {base_dir_path}")


if __name__ == "__main__":
    args = parse_args()

    learning_rates = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1]
    batch_sizes = [8, 16, 32, 64, 128, 256, 512]

    create_config_files(
        base_dir=args.base_dir,
        learning_rates=learning_rates,
        batch_sizes=batch_sizes,
        total_epochs=args.total_epochs,
        aggregation_per_epoch=args.aggregation_per_epoch,
        iterations=args.iterations,
    )
