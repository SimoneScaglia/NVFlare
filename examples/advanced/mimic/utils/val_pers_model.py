import os
import argparse
import sys
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from datetime import datetime

sys.path.append(os.path.join(os.path.dirname(__file__), '../networks'))
from mimic_nets import CNN, get_opt, get_metrics

def load_and_preprocess_data(path, scaler=None, fit=False):
    data = pd.read_csv(path)
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    if scaler is not None:
        if fit:
            X = scaler.fit_transform(X)
        else:
            X = scaler.transform(X)
    return X, y

def personalized_fine_tune(data_path, models_path, iteration, num_clients):
    results_dir = f'/tmp/nvflare/results/1host_{num_clients}nodes/swarm_results'
    os.makedirs(results_dir, exist_ok=True)
    results_file = os.path.join(results_dir, f'aucs.csv')

    if os.path.exists(results_file):
        results_df = pd.read_csv(results_file)
    else:
        results_df = pd.DataFrame(columns=['datetime', 'user', 'splits', 'auc', 'iteration'])

    scaler = StandardScaler()
    X_test, y_test = load_and_preprocess_data(data_path, scaler, fit=True)
    input_dim = X_test.shape[1]

    for j in range(1, num_clients + 1):
        model_path = os.path.join(models_path, f'site-{j}', 'simulate_job', f'app_site-{j}', f'site-{j}.weights.h5')
        train_data_path = f'/tmp/mimic_data/site-{j}.csv'

        if not os.path.exists(model_path):
            print(f"[Client {j}] Model not found: {model_path}")
            continue

        try:
            X_train, y_train = load_and_preprocess_data(train_data_path, scaler, fit=True)

            model = CNN(input_dim=input_dim)
            model.build((None, input_dim))
            model.load_weights(model_path)

            model.compile(optimizer=get_opt(), loss='binary_crossentropy', metrics=get_metrics())
            model.fit(X_train, y_train, epochs=5, batch_size=64, verbose=0)

            y_pred = model.predict(X_test).flatten()
            auc = roc_auc_score(y_test, y_pred)

            new_row = {
                'datetime': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'user': j,
                'splits': '/',
                'auc': auc,
                'iteration': iteration
            }
            results_df = pd.concat([results_df, pd.DataFrame([new_row])], ignore_index=True)
            print(f"[Client {j}] Personalized AUC: {auc:.4f}")

        except Exception as e:
            print(f"[Client {j}] Error during fine-tuning: {str(e)}")

    results_df.to_csv(results_file, index=False)
    print(f"\nResults saved to {results_file}")

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    parser = argparse.ArgumentParser(description="Script per validare modelli personalizzati (fine-tuned).")
    parser.add_argument('iteration', type=int, help="Iterazione.")
    parser.add_argument('num_clients', type=int, help="Numero di client.")

    args = parser.parse_args()
    data_path = '../dataset/mimiciv_test.csv'
    models_path = f'/tmp/nvflare/mimic_swarm_{args.num_clients}'

    personalized_fine_tune(data_path, models_path, args.iteration, args.num_clients)

if __name__ == "__main__":
    main()
