#!/usr/bin/env python3
"""
Local training: each node trains its own model independently on its local data.
Evaluates each local model on the shared test set and saves metrics to CSV.

Usage:
    python run-local.py <num_nodes> <configuration> <iteration> <lr> <bs>

Example:
    python run-local.py 5 10nodes 3 0.0025 512
    -> Trains 5 independent local models using node1_3.csv .. node5_3.csv
       from datasets/mimic_iv/mimiciv_same_size/10nodes/
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import random
from datetime import datetime

import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, precision_score, recall_score

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['PYTHONHASHSEED'] = str(SEED)
tf.config.experimental.enable_op_determinism()


def get_net(input_dim):
    """Build the FCN model (same architecture as mimic_nets.py)."""
    kernel_init = tf.keras.initializers.GlorotUniform(seed=SEED)
    bias_init = tf.keras.initializers.Zeros()
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(16, activation="relu", input_shape=(input_dim,),
                              kernel_initializer=kernel_init, bias_initializer=bias_init),
        tf.keras.layers.Dense(16, activation="relu",
                              kernel_initializer=kernel_init, bias_initializer=bias_init),
        tf.keras.layers.Dense(16, activation="relu",
                              kernel_initializer=kernel_init, bias_initializer=bias_init),
        tf.keras.layers.Dense(1, activation="sigmoid",
                              kernel_initializer=kernel_init, bias_initializer=bias_init),
    ])
    return model


def get_metrics():
    return [
        tf.keras.metrics.AUC(name='auc', curve='ROC', num_thresholds=1000),
        tf.keras.metrics.AUC(name='auprc', curve='PR', num_thresholds=1000),
        tf.keras.metrics.BinaryAccuracy(name='accuracy'),
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall'),
    ]


def main():
    parser = argparse.ArgumentParser(description="Local training for same-size experiments")
    parser.add_argument('num_nodes', type=int, help="Number of nodes to use (2..max)")
    parser.add_argument('configuration', type=str, help="Configuration folder name (e.g. 10nodes)")
    parser.add_argument('iteration', type=int, help="Iteration index (0-9)")
    parser.add_argument('lr', type=float, help="Learning rate")
    parser.add_argument('bs', type=int, help="Batch size")
    args = parser.parse_args()

    num_nodes = args.num_nodes
    configuration = args.configuration
    iteration = args.iteration
    lr = args.lr
    bs = args.bs
    epochs = 25

    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "datasets", "mimic_iv", "mimiciv_same_size", configuration)
    test_path = os.path.join(script_dir, "datasets", "mimic_iv", "test.csv")

    # Results file
    results_dir = os.path.join(script_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    results_file = os.path.join(results_dir, "local_results.csv")

    # Load test data
    test_df = pd.read_csv(test_path)
    X_test = test_df.iloc[:, :-1].values.astype(np.float32)
    y_test = test_df.iloc[:, -1].values.astype(np.float32)
    input_dim = X_test.shape[1]

    # Load or create results dataframe
    if os.path.exists(results_file):
        results_df = pd.read_csv(results_file)
    else:
        results_df = pd.DataFrame(columns=[
            'datetime', 'configuration', 'num_nodes', 'node_id',
            'iteration', 'lr', 'batch_size', 'epochs',
            'loss', 'auc', 'auprc', 'accuracy', 'precision', 'recall'
        ])

    # Train each node's model independently
    for node_id in range(1, num_nodes + 1):
        train_file = os.path.join(data_dir, f"node{node_id}_{iteration}.csv")
        if not os.path.exists(train_file):
            print(f"WARNING: {train_file} does not exist, skipping node {node_id}")
            continue

        print(f"[Local] Config={configuration} Nodes={num_nodes} Iter={iteration} Node={node_id}")

        # Load training data
        train_df = pd.read_csv(train_file)
        X_train = train_df.iloc[:, :-1].values.astype(np.float32)
        y_train = train_df.iloc[:, -1].values.astype(np.float32)

        # Build and compile model
        tf.keras.backend.clear_session()
        random.seed(SEED)
        np.random.seed(SEED)
        tf.random.set_seed(SEED)

        model = get_net(input_dim)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
            loss='binary_crossentropy',
            metrics=get_metrics()
        )

        # Train
        model.fit(X_train, y_train, epochs=epochs, batch_size=bs, verbose=0)

        # Evaluate on test set
        metrics = model.evaluate(X_test, y_test, verbose=0, return_dict=True)

        new_row = {
            'datetime': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'configuration': configuration,
            'num_nodes': num_nodes,
            'node_id': node_id,
            'iteration': iteration,
            'lr': lr,
            'batch_size': bs,
            'epochs': epochs,
            'loss': metrics['loss'],
            'auc': metrics['auc'],
            'auprc': metrics['auprc'],
            'accuracy': metrics['accuracy'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
        }

        results_df = pd.concat([results_df, pd.DataFrame([new_row])], ignore_index=True)
        print(f"  -> AUC={metrics['auc']:.4f}  AUPRC={metrics['auprc']:.4f}  "
              f"Acc={metrics['accuracy']:.4f}  Loss={metrics['loss']:.4f}")

    # Save results
    results_df.to_csv(results_file, index=False)
    print(f"Results saved to {results_file}")


if __name__ == "__main__":
    main()
