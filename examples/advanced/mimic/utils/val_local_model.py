import os
import argparse
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import random
import tensorflow as tf

random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
os.environ['PYTHONHASHSEED'] = str(42)

sys.path.append(os.path.join(os.path.dirname(__file__), '../networks'))
from mimic_nets import get_net, get_opt, get_metrics

def load_and_preprocess_data(path):
    data = pd.read_csv(path)
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values

    return X, y

def validate_local_models(data_path, iteration, num_clients, weights):
    if not np.isclose(sum(weights), 100.0, atol=1e-1):
        is_data = True
        results_dir = f'/tmp/nvflare/results/1host_{num_clients}nodes_data/swarm_results'
        os.makedirs(results_dir, exist_ok=True)
        results_file = os.path.join(results_dir, f'local_aucs.csv')
    else:
        is_data = False
        weights = weights[::-1]
        results_dir = f'/tmp/nvflare/results/1host_{num_clients}nodes/swarm_results'
        os.makedirs(results_dir, exist_ok=True)
        results_file = os.path.join(results_dir, f'local_aucs.csv')
    
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

    first_client = 2 if is_data else 1
    for j in range(first_client, num_clients + first_client):
        weight = weights[j - first_client] if weights else 100.0 / num_clients
        if weight.is_integer():
            weight = int(weight)
        train_data_path = f'../dataset/mimiciii_{num_clients if is_data else num_clients - 1}_nodes/{j}_{weight}_{iteration}.csv'

        X_test, y_test = load_and_preprocess_data(data_path)

        try:
            scaler = StandardScaler()
            X_train, y_train = load_and_preprocess_data(train_data_path)
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            # Build and compile model
            model = get_net(X_test.shape[1])
            model.compile(optimizer=get_opt(), loss='binary_crossentropy', metrics=get_metrics())

            # Train model
            model.fit(X_train, y_train, epochs=25, batch_size=64, verbose=0)

            # Predict and evaluate
            metrics = model.evaluate(X_test, y_test, verbose=0, return_dict=True)

            new_row = {
                'datetime': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'user': j,
                'splits': weight,
                'loss': metrics['loss'],
                'auc': metrics['auc'],
                'auprc': metrics['auprc'],
                'accuracy': metrics['accuracy'],
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'iteration': iteration
            }
            results_df = pd.concat([results_df, pd.DataFrame([new_row])], ignore_index=True)
            print(f"[Client {j}] AUC: {metrics['auc']:.4f}")

        except Exception as e:
            print(f"[Client {j}] Error: {str(e)}")

    results_df.to_csv(results_file, index=False)
    print(f"\nResults saved to {results_file}")

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    parser = argparse.ArgumentParser(description="Script per valutare il modello locale.")
    parser.add_argument('iteration', type=int, help="Iterazione.")
    parser.add_argument('num_clients', type=int, help="Numero di client.")
    parser.add_argument('--weights', type=float, nargs='+', help="Percentuali di dati per ciascun client")

    args = parser.parse_args()

    data_path = '../dataset/mimiciii_test.csv'
    validate_local_models(data_path, args.iteration, args.num_clients, args.weights)

if __name__ == "__main__":
    main()
