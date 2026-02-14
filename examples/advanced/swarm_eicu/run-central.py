#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import gc
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import argparse
import json

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.metrics import AUC, BinaryAccuracy, Precision, Recall

# seeds
np.random.seed(42)
tf.keras.utils.set_random_seed(42)

LABEL_COL_INDEX = -1
NODES = None
DATA_DIR = None

# parametri fissi
EPOCHS = None
BATCH_SIZE = None
LEARNING_RATE = None

# fixed output values
OUTPUT_USER = "central"
OUTPUT_SPLITS = 100
OUTPUT_COLUMNS = ["datetime", "user", "splits", "loss", "auc", "auprc", "accuracy", "precision", "recall", "iteration"]

# ==========================
# DEFINIZIONE MODELLO FCN
# ==========================
def build_fcn(input_dim: int) -> keras.Model:
    """Costruisce una Fully Connected Network come in mimic_nets.py"""
    kernel_initializer = tf.keras.initializers.GlorotUniform(seed=42)
    bias_initializer = tf.keras.initializers.Zeros()

    model = keras.Sequential([
        keras.layers.Dense(16, activation="relu", input_shape=(input_dim,), kernel_initializer=kernel_initializer, bias_initializer=bias_initializer),
        keras.layers.Dense(16, activation="relu", kernel_initializer=kernel_initializer, bias_initializer=bias_initializer),
        keras.layers.Dense(16, activation="relu", kernel_initializer=kernel_initializer, bias_initializer=bias_initializer),
        keras.layers.Dense(1, activation="sigmoid", kernel_initializer=kernel_initializer, bias_initializer=bias_initializer)
    ])
    return model


def get_optimizer():
    return tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)


def get_metrics():
    return [
        AUC(name='auc', curve='ROC', num_thresholds=1000),
        AUC(name='auprc', curve='PR', num_thresholds=1000),
        BinaryAccuracy(name='accuracy'),
        Precision(name='precision'),
        Recall(name='recall')
    ]


# ==========================
# FUNZIONI DI UTILITÀ
# ==========================
def prepare_xy(df: pd.DataFrame, feature_columns=None):
    if feature_columns is None:
        X_cols = [c for c in df.columns if c != df.columns[LABEL_COL_INDEX]]
    else:
        X_cols = feature_columns
    missing = [c for c in X_cols if c not in df.columns]
    for c in missing:
        df[c] = 0.0
    X = df[X_cols].astype(np.float32).to_numpy()
    y = df[df.columns[LABEL_COL_INDEX]].astype(np.float32).to_numpy().reshape(-1, 1)
    return X, y, X_cols


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def append_result_csv(path: Path, row: dict):
    full = {col: row.get(col, None) for col in OUTPUT_COLUMNS}
    df_row = pd.DataFrame([full], columns=OUTPUT_COLUMNS)
    if path.exists():
        df_row.to_csv(path, mode='a', header=False, index=False)
    else:
        df_row.to_csv(path, mode='w', header=True, index=False)


def current_datetime_rome_iso():
    return datetime.now(ZoneInfo("Europe/Rome")).isoformat()


# ==========================
# TRAIN + EVAL
# ==========================
HOSPITAL_IDS = [1, 2, 3, 4, 5]  # Hospital IDs cablati

# Modifica della funzione per caricare e concatenare i dati

def load_and_concat_datasets(data_dir: Path, iteration: int, dataset_type: str) -> pd.DataFrame:
    """
    Carica e concatena i file di train o test dei singoli nodi, mescolandoli in base all'iterazione.

    Args:
        data_dir (Path): Directory dei dati.
        iteration (int): Iterazione corrente per mescolare i dati.
        dataset_type (str): Tipo di dataset ('train' o 'test').

    Returns:
        pd.DataFrame: DataFrame concatenato e mescolato.
    """
    dfs = []
    for hospital_id in HOSPITAL_IDS:
        file_path = data_dir / f"{hospital_id}_{dataset_type}.csv"
        if not file_path.exists():
            raise FileNotFoundError(f"File mancante: {file_path}")
        df = pd.read_csv(file_path)
        dfs.append(df)

    concatenated_df = pd.concat(dfs, axis=0, ignore_index=True)
    return concatenated_df.sample(frac=1, random_state=iteration).reset_index(drop=True)

def train_and_eval(x_nodes: int, iteration_y: int, epochs: int, batch_size: int, verbose=0):
    print(f"\n=== x_nodes={x_nodes}, iteration={iteration_y} ===")

    # Caricamento e concatenazione dei dataset
    df_train = load_and_concat_datasets(DATA_DIR, iteration_y, "train")
    df_test = load_and_concat_datasets(DATA_DIR, iteration_y, "test")

    X_train, y_train, feature_cols = prepare_xy(df_train)
    X_test, y_test, _ = prepare_xy(df_test, feature_columns=feature_cols)

    input_dim = X_train.shape[1]
    print(f"Train samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}, input_dim={input_dim}")

    model = build_fcn(input_dim)
    model.compile(
        optimizer=get_optimizer(),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=get_metrics()
    )

    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=verbose, shuffle=True)
    eval_results = model.evaluate(X_test, y_test, batch_size=batch_size, verbose=0, return_dict=True)

    tf.keras.backend.clear_session()
    del model
    gc.collect()

    return eval_results

def parse_args():
    parser = argparse.ArgumentParser(description="Run global learner with configurable parameters.")
    parser.add_argument("-c", "--config", type=str, required=True, help="Path to the configuration JSON file.")
    return parser.parse_args()

# ==========================
# MAIN LOOP
# ==========================
def main():
    args = parse_args()

    # Load configuration from JSON file
    with open(args.config, "r") as f:
        config = json.load(f)

    global NODES, DATA_DIR, TEST_FILE, EPOCHS, BATCH_SIZE, LEARNING_RATE
    NODES = config["num_nodes"]
    DATA_DIR = Path(config["data_directory"])
    EPOCHS = 25
    BATCH_SIZE = config["hyperparameters"]["batch_size"]
    LEARNING_RATE = config["hyperparameters"]["learning_rate"]

    out_dir = Path(config["results_directory"])
    ensure_dir(out_dir)
    out_file = out_dir / "central_results.csv"

    results = train_and_eval(x_nodes=config["num_nodes"], iteration_y=config["iteration"], epochs=EPOCHS, batch_size=BATCH_SIZE)

    row = {
        "datetime": current_datetime_rome_iso(),
        "user": OUTPUT_USER,
        "splits": OUTPUT_SPLITS,
        'loss': results['loss'],
        'auc': results['auc'],
        'auprc': results['auprc'],
        'accuracy': results['accuracy'],
        'precision': results['precision'],
        'recall': results['recall'],
        "iteration": config["iteration"]
    }
    append_result_csv(out_file, row)
    print(f"Salvato risultato in {out_file}")


if __name__ == "__main__":
    main()
