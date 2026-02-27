#!/usr/bin/env python3
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow import keras

# --- Reproducibility
def set_seed(seed=42):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    import random
    random.seed(seed)

set_seed(42)
warnings.filterwarnings("ignore")

# --- Model builder (uses the architecture you provided)
def build_model(input_shape):
    kernel_initializer = tf.keras.initializers.GlorotUniform(seed=42)
    bias_initializer = tf.keras.initializers.Zeros()
    model = keras.Sequential([
        keras.layers.Dense(16, activation="relu", input_shape=(input_shape,), kernel_initializer=kernel_initializer, bias_initializer=bias_initializer),
        keras.layers.Dense(16, activation="relu", kernel_initializer=kernel_initializer, bias_initializer=bias_initializer),
        keras.layers.Dense(16, activation="relu", kernel_initializer=kernel_initializer, bias_initializer=bias_initializer),
        keras.layers.Dense(1, activation="sigmoid", kernel_initializer=kernel_initializer, bias_initializer=bias_initializer)
    ])
    return model

# --- Single dataset run (train on one train_csv, evaluate on test_df)
def run_on_dataset(train_csv, test_df, batch_sizes, lrs, epochs, results_list):
    df = pd.read_csv(train_csv)
    if df.shape[0] == 0:
        print(f"[WARN] {train_csv} empty, skipping.")
        return

    # last column is target
    X_train = df.iloc[:, :-1].copy()
    y_train = df.iloc[:, -1].copy().astype(int)

    # test features/target
    X_test = test_df.iloc[:, :-1].copy()
    y_test = test_df.iloc[:, -1].copy().astype(int)

    # Fill missing numeric NaNs
    X_train = X_train.fillna(X_train.mean())
    X_test = X_test.fillna(X_train.mean())

    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    input_dim = X_train_scaled.shape[1]

    for batch_size in batch_sizes:
        for lr in lrs:
            tf.keras.backend.clear_session()
            set_seed(42)  # riproducibilità
            model = build_model(input_dim)
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                loss='binary_crossentropy',
                metrics=[keras.metrics.AUC(name='auc')]
            )

            # Training
            h = model.fit(
                X_train_scaled, y_train,
                validation_data=(X_test_scaled, y_test),
                epochs=epochs,
                batch_size=batch_size,
                verbose=0
            )

            # Prendi l'ultimo valore di AUC sul validation set
            auc = h.history['val_auc'][-1]

            results_list.append({
                "dataset": os.path.basename(train_csv),
                "train_path": os.path.abspath(train_csv),
                "batch_size": int(batch_size),
                "learning_rate": float(lr),
                "epochs": int(epochs),
                "auc": float(auc)
            })

            tf.keras.backend.clear_session()

    # End dataset

def plot_heatmap_for_dataset(results_df, dataset_name, out_dir):
    """results_df: subset for the dataset with columns batch_size, learning_rate, auc"""
    pivot = results_df.pivot(index="batch_size", columns="learning_rate", values="auc")
    # sort index/columns
    pivot = pivot.sort_index(ascending=True)
    pivot = pivot.reindex(sorted(pivot.columns), axis=1)

    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot, annot=True, fmt=".4f", linewidths=.5, cbar_kws={'label': 'AUC'}, vmin=0.0, vmax=1.0)
    plt.title(f"AUC heatmap - {dataset_name}")
    plt.xlabel("learning_rate")
    plt.ylabel("batch_size")
    fname = os.path.join(out_dir, f"heatmap_{dataset_name}.png")
    plt.tight_layout()
    plt.savefig(fname, dpi=200)
    plt.close()
    return fname

def main():
    dir = "80nodes"
    # Defaults from your request
    batch_sizes = [32, 64, 128]
    lrs = [0.0100, 0.0250, 0.0500, 0.0750, 0.0010, 0.0025, 0.0050]
    epochs = 25

    # Validate paths
    data_folder = Path(dir)
    if not data_folder.is_dir():
        raise ValueError(f"{data_folder} non è una cartella valida.")

    test_csv = Path("../25k_test.csv")
    if not test_csv.is_file():
        raise ValueError(f"Test CSV {test_csv} non trovato.")

    out_dir = Path(f"{dir}_results")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Read test once
    test_df = pd.read_csv(test_csv)

    # Find csv files in folder
    train_files = sorted([str(p) for p in data_folder.glob("*.csv")])[:10]
    if len(train_files) == 0:
        raise ValueError(f"Nessun .csv trovato in {data_folder}")

    results = []
    total_runs = len(train_files) * len(batch_sizes) * len(lrs)

    print(f"Found {len(train_files)} train files. Total runs: {total_runs}. Epochs fixed = {epochs}.")
    for train_csv in train_files:
        print(f"\n--- Processing {os.path.basename(train_csv)} ---")
        run_on_dataset(train_csv, test_df, batch_sizes, lrs, epochs, results)
        # Save heatmap for this dataset
        df_subset = pd.DataFrame(results)
        df_this = df_subset[df_subset["dataset"] == os.path.basename(train_csv)]
        if not df_this.empty:
            heatmap_path = plot_heatmap_for_dataset(df_this, Path(train_csv).stem, str(out_dir))
            print(f"Saved heatmap: {heatmap_path}")

    # Save overall results csv
    results_df = pd.DataFrame(results)
    csv_out = os.path.join(out_dir, f"{dir}_results.csv")
    results_df.to_csv(csv_out, index=False)
    
    mean_df = (
        results_df
        .groupby(["batch_size", "learning_rate"], as_index=False)["auc"]
        .mean()
    )

    pivot = mean_df.pivot(index="batch_size", columns="learning_rate", values="auc")
    pivot = pivot.sort_index(ascending=True)
    pivot = pivot.reindex(sorted(pivot.columns), axis=1)

    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot, annot=True, fmt=".4f", linewidths=.5, cbar_kws={'label': 'Mean AUC'}, vmin=0.0, vmax=1.0)
    plt.title(f"Mean AUC heatmap across all datasets ({dir})")
    plt.xlabel("learning_rate")
    plt.ylabel("batch_size")
    fname = os.path.join(out_dir, f"heatmap_mean_{dir}.png")
    plt.tight_layout()
    plt.savefig(fname, dpi=200)
    plt.close()
    print(f"Saved mean heatmap: {fname}")

    print("Done.")

main()
