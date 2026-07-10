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
    parser.add_argument("--fedprox-mu", type=float, default=1e-5, help="FedProx proximal term strength (default used when generating single-mu sets)")
    parser.add_argument(
        "--evaluation-mode",
        choices=["all", "entire", "separate"],
        default="all",
        help="Generate configs for all modes or only one evaluation mode",
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


def create_config_files(base_dir, total_epochs, aggregation_per_epoch=5, iterations=5, evaluation_mode="all"):
    if total_epochs <= 0:
        raise ValueError("total_epochs must be > 0")
    if aggregation_per_epoch <= 0:
        raise ValueError("aggregation_per_epoch must be > 0")
    if total_epochs % aggregation_per_epoch != 0:
        raise ValueError("total_epochs must be divisible by aggregation_per_epoch")

    base_config = {
        "experiment_name": "",
        "fedproxloss_mu": 0.0,
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

    eicu_variants_hyperparamers = {
        "data_20k_entire":              [0.0005, 32],
        "data_20k_fixed_rows_entire":   [0.0001,  8],
        "data_20k_fixed_rows_separate": [0.0001,  8],
        "data_20k_separate":            [0.0005, 32],
        "data_entire":                  [0.0001,  8],
        "data_fixed_rows_entire":       [0.0001,  8],
        "data_fixed_rows_separate":     [0.0001,  8],
        "data_separate":                [0.0001,  8],
    }

    base_dir_path = Path(base_dir)
    base_dir_path.mkdir(parents=True, exist_ok=True)

    # Remove stale JSON files from previous generations.
    for old_json in base_dir_path.glob("*.json"):
        old_json.unlink()

    swarm_eicu_dir = Path(__file__).resolve().parent
    num_rounds = total_epochs // aggregation_per_epoch
    # We'll generate configs for multiple fedprox mu values (if desired).
    fedprox_values = [10, 1, 0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7]

    created = 0
    for eicu_variant in eicu_variants_hyperparamers.keys():
        variant_mode = "entire" if "_entire" in eicu_variant else "separate"
        if evaluation_mode != "all" and variant_mode != evaluation_mode:
            continue

        data_dir = swarm_eicu_dir / "datasets" / "eicu" / eicu_variant.removesuffix("_entire").removesuffix("_separate")
        lr, bs = eicu_variants_hyperparamers[eicu_variant]
        if not data_dir.exists():
            raise FileNotFoundError(f"Dataset directory not found: {data_dir}")

        dataset_ids = discover_dataset_ids(data_dir)
        num_nodes = len(dataset_ids)
        node_weights = build_uniform_node_weights(num_nodes)

        results_subdir = f"eicu_{eicu_variant}_testset"
        min_responses = num_nodes

        for iteration in range(iterations):
            for fedprox in fedprox_values:
                config = copy.deepcopy(base_config)
                config["fedproxloss_mu"] = fedprox
                # Keep results directory identical across fedprox variants
                config["experiment_name"] = (
                    f"eicu_{eicu_variant}_{total_epochs}_{iteration}_lr{lr:.5f}_bs{bs}"
                )
                config["num_nodes"] = num_nodes
                config["node_weights"] = node_weights
                config["min_responses_for_aggregation"] = min_responses
                config["num_aggregation_rounds"] = num_rounds
                config["aggregation_per_epoch"] = aggregation_per_epoch
                config["hyperparameters"]["learning_rate"] = lr
                config["hyperparameters"]["batch_size"] = bs
                config["data_directory"] = f"datasets/eicu/{eicu_variant.removesuffix('_entire').removesuffix('_separate')}/"
                config["dataset_ids"] = dataset_ids
                config["evaluation_mode"] = variant_mode
                config["results_directory"] = (
                    f"fedprox_new_results/hp_fixed_no_best_model/{results_subdir}/{total_epochs}_lr{lr:.5f}_bs{bs}/"
                )
                config["iteration"] = iteration
                # Use fedprox value in the file name so configs are unique
                fedprox_str = f"{fedprox:.0e}".replace("-0", "")
                file_name = (
                    f"eicu_{eicu_variant}_{total_epochs}_{iteration}_lr{lr:.5f}_bs{bs}_fedprox{fedprox_str}"
                ).replace("0.", "0-") + ".json"
                file_path = base_dir_path / file_name

                with open(file_path, "w", encoding="utf-8") as f:
                    json.dump(config, f, indent=4)
                created += 1

    print(f"Generated {created} config files in: {base_dir_path}")


if __name__ == "__main__":
    args = parse_args()

    create_config_files(
        base_dir=args.base_dir,
        total_epochs=args.total_epochs,
        aggregation_per_epoch=args.aggregation_per_epoch,
        iterations=args.iterations,
        evaluation_mode=args.evaluation_mode,
    )
