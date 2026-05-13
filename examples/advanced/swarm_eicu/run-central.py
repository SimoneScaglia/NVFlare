#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import gc
import json
import re
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.metrics import AUC, BinaryAccuracy, Precision, Recall

# Reproducibility
np.random.seed(42)
tf.keras.utils.set_random_seed(42)

LABEL_COL_INDEX = -1
ID_TRAIN_RE = re.compile(r"^(\d+)_train\.csv$")

OUTPUT_COLUMNS = [
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


def parse_args():
    parser = argparse.ArgumentParser(description="Run central baseline for eICU experiments.")
    parser.add_argument("-c", "--config", type=str, required=True, help="Path to experiment JSON config")
    return parser.parse_args()


def current_datetime_rome_iso():
    return datetime.now(ZoneInfo("Europe/Rome")).isoformat()


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def append_result_csv(path: Path, row: dict):
    ordered = {col: row.get(col, None) for col in OUTPUT_COLUMNS}
    row_df = pd.DataFrame([ordered], columns=OUTPUT_COLUMNS)
    if path.exists():
        row_df.to_csv(path, mode="a", header=False, index=False)
    else:
        row_df.to_csv(path, mode="w", header=True, index=False)


def build_fcn(input_dim: int) -> keras.Model:
    kernel_initializer = tf.keras.initializers.GlorotUniform(seed=42)
    bias_initializer = tf.keras.initializers.Zeros()

    model = keras.Sequential(
        [
            keras.layers.Dense(
                16,
                activation="relu",
                input_shape=(input_dim,),
                kernel_initializer=kernel_initializer,
                bias_initializer=bias_initializer,
            ),
            keras.layers.Dense(16, activation="relu", kernel_initializer=kernel_initializer, bias_initializer=bias_initializer),
            keras.layers.Dense(16, activation="relu", kernel_initializer=kernel_initializer, bias_initializer=bias_initializer),
            keras.layers.Dense(1, activation="sigmoid", kernel_initializer=kernel_initializer, bias_initializer=bias_initializer),
        ]
    )
    return model


def get_optimizer(learning_rate: float):
    return tf.keras.optimizers.Adam(learning_rate=learning_rate)


def get_metrics():
    return [
        AUC(name="auc", curve="ROC", num_thresholds=1000),
        AUC(name="auprc", curve="PR", num_thresholds=1000),
        BinaryAccuracy(name="accuracy"),
        Precision(name="precision"),
        Recall(name="recall"),
    ]


def prepare_xy(df: pd.DataFrame, feature_columns=None):
    if feature_columns is None:
        x_cols = [c for c in df.columns if c != df.columns[LABEL_COL_INDEX]]
    else:
        x_cols = feature_columns

    missing = [c for c in x_cols if c not in df.columns]
    for c in missing:
        df[c] = 0.0

    x = df[x_cols].astype(np.float32).to_numpy()
    y = df[df.columns[LABEL_COL_INDEX]].astype(np.float32).to_numpy().reshape(-1, 1)
    return x, y, x_cols


def discover_dataset_ids(data_dir: Path):
    ids = []
    for file_path in sorted(data_dir.glob("*_train.csv")):
        match = ID_TRAIN_RE.match(file_path.name)
        if not match:
            continue
        dataset_id = int(match.group(1))
        test_path = data_dir / f"{dataset_id}_test.csv"
        if test_path.exists():
            ids.append(dataset_id)

    if not ids:
        raise ValueError(f"No <id>_train.csv and <id>_test.csv pairs found in {data_dir}")

    return ids


def load_split_df(data_dir: Path, dataset_id: int, split: str):
    path = data_dir / f"{dataset_id}_{split}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing dataset file: {path}")
    return pd.read_csv(path)


def load_concat_split_df(data_dir: Path, dataset_ids, split: str, shuffle_seed: int):
    frames = [load_split_df(data_dir, dataset_id, split) for dataset_id in dataset_ids]
    df = pd.concat(frames, axis=0, ignore_index=True)
    return df.sample(frac=1, random_state=shuffle_seed).reset_index(drop=True)


def evaluate_checkpoints(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    learning_rate: float,
    batch_size: int,
    epochs: int,
    out_file: Path,
    user,
    splits: float,
    iteration: int,
    eval_every: int = 5,
    verbose=0,
):
    x_train, y_train, feature_cols = prepare_xy(df_train)
    x_test, y_test, _ = prepare_xy(df_test, feature_columns=feature_cols)

    input_dim = x_train.shape[1]

    model = build_fcn(input_dim)
    model.compile(
        optimizer=get_optimizer(learning_rate),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=get_metrics(),
    )

    checkpoints = list(range(eval_every, epochs + 1, eval_every))
    if not checkpoints or checkpoints[-1] != epochs:
        checkpoints.append(epochs)

    prev_epoch = 0
    for target_epoch in checkpoints:
        model.fit(
            x_train,
            y_train,
            initial_epoch=prev_epoch,
            epochs=target_epoch,
            batch_size=batch_size,
            verbose=verbose,
            shuffle=True,
        )

        metrics = model.evaluate(x_test, y_test, batch_size=batch_size, verbose=0, return_dict=True)

        row = {
            "datetime": current_datetime_rome_iso(),
            "user": user,
            "splits": splits,
            "loss": metrics["loss"],
            "auc": metrics["auc"],
            "auprc": metrics["auprc"],
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "iteration": iteration,
            "epoch": target_epoch,
        }
        append_result_csv(out_file, row)
        prev_epoch = target_epoch

    tf.keras.backend.clear_session()
    del model
    gc.collect()


def run_central(config: dict, data_dir: Path, dataset_ids, out_file: Path, epochs: int, batch_size: int, learning_rate: float):
    iteration = int(config.get("iteration", 0))

    df_train = load_concat_split_df(data_dir, dataset_ids, "train", shuffle_seed=iteration)
    df_test = load_concat_split_df(data_dir, dataset_ids, "test", shuffle_seed=iteration)

    evaluate_checkpoints(
        df_train=df_train,
        df_test=df_test,
        learning_rate=learning_rate,
        batch_size=batch_size,
        epochs=epochs,
        out_file=out_file,
        user="central",
        splits=100,
        iteration=iteration,
        eval_every=5,
    )


def main():
    args = parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = json.load(f)

    script_dir = Path(__file__).resolve().parent

    data_dir = Path(config["data_directory"])
    if not data_dir.is_absolute():
        data_dir = (script_dir / data_dir).resolve()

    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")

    dataset_ids = config.get("dataset_ids") or discover_dataset_ids(data_dir)
    dataset_ids = [int(x) for x in dataset_ids]
    if not dataset_ids:
        raise ValueError("dataset_ids cannot be empty")

    num_nodes = int(config.get("num_nodes", len(dataset_ids)))
    if num_nodes != len(dataset_ids):
        raise ValueError(f"num_nodes ({num_nodes}) does not match dataset_ids count ({len(dataset_ids)})")

    epochs = int(config.get("num_aggregation_rounds", 1) * config.get("aggregation_per_epoch", 1))
    if epochs <= 0:
        raise ValueError("Computed epochs must be > 0")

    batch_size = int(config["hyperparameters"]["batch_size"])
    learning_rate = float(config["hyperparameters"]["learning_rate"])

    results_dir = Path(config["results_directory"])
    if not results_dir.is_absolute():
        results_dir = (script_dir / results_dir).resolve()
    ensure_dir(results_dir)
    out_file = results_dir / "central_results.csv"

    eval_mode = str(config.get("evaluation_mode", "entire")).strip().lower()

    print(f"Running central baseline for config: {args.config}")
    print(f"  mode={eval_mode}, ids={dataset_ids}, epochs={epochs}, bs={batch_size}, lr={learning_rate}")
    print(f"  output={out_file}")

    run_central(
        config=config,
        data_dir=data_dir,
        dataset_ids=dataset_ids,
        out_file=out_file,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
    )

    print(f"Central results saved to {out_file}")


if __name__ == "__main__":
    main()
