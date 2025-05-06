import os
import argparse
import sys
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from datetime import datetime

sys.path.append(os.path.join(os.path.dirname(__file__), '../networks'))
from mimic_nets import CNN

DEFAULT_ITER=5

def validate_global_models(data_path, models_path, iteration, num_clients, weights):
    if not np.isclose(sum(weights), 100.0, atol=1e-1):
        results_dir = f'/tmp/nvflare/results/1host_{num_clients}nodes_data/swarm_results'
        os.makedirs(results_dir, exist_ok=True)
        results_file = os.path.join(results_dir, f'aucs.csv')
    else:
        results_dir = f'/tmp/nvflare/results/1host_{num_clients}nodes/swarm_results'
        os.makedirs(results_dir, exist_ok=True)
        results_file = os.path.join(results_dir, f'aucs.csv')
    
    data = pd.read_csv(data_path)
    X_test = data.iloc[:, :-1].values
    y_test = data.iloc[:, -1].values
    
    input_dim = X_test.shape[1]
    
    if os.path.exists(results_file):
        results_df = pd.read_csv(results_file)
    else:
        results_df = pd.DataFrame(columns=['datetime', 'user', 'split', 'auc', 'iteration'])
    
    for j in range(1, num_clients + 1):
        model_path = os.path.join(models_path, f'site-{j}', 'simulate_job', f'app_site-{j}', f'site-{j}.weights.h5')
        
        if not os.path.exists(model_path):
            print(f"Model not found: {model_path}")
            continue
            
        try:
            model = CNN(input_dim=input_dim)
            model.build((None, input_dim))
            model.load_weights(model_path)
            
            y_pred = model.predict(X_test).flatten()
            auc = roc_auc_score(y_test, y_pred)
            
            weight = weights[j - 1] if weights else 100.0/num_clients
            new_row = {'datetime': datetime.now(), 'user': j, 'split': weight, 'auc': auc, 'iteration': iteration}
            
            results_df = pd.concat([results_df, pd.DataFrame([new_row])], ignore_index=True)
        except Exception as e:
            print(f"Error evaluating model at {model_path}: {str(e)}")

    results_df.to_csv(results_file, index=False)
    print(f"Results saved to {results_file}")

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    parser = argparse.ArgumentParser(description="Script per validare il modello globale.")
    parser.add_argument('iteration', type=int, help="Iterazione.")
    parser.add_argument('num_clients', type=int, help="Numero di client.")
    parser.add_argument('--weights', type=float, nargs='+', help="Percentuali di dati per ciascun client (devono sommare a 100.0)")

    args = parser.parse_args()

    data_path = '../dataset/mimiciv_test.csv'
    models_path = f'/tmp/nvflare/mimic_swarm_{args.num_clients}'

    validate_global_models(data_path, models_path, args.iteration, args.num_clients, args.weights)

if __name__ == "__main__":
    main()
