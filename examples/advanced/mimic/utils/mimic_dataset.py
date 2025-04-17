import numpy as np
import pandas as pd
import os
import shutil
import argparse
from sklearn.model_selection import StratifiedShuffleSplit

def create_site_files(data_path, idx_root, num_clients, stratify=False, client_weights=None):
    # Carica il dataset
    df = pd.read_csv(data_path)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    total_size = len(X)

    # Validazioni
    if num_clients > total_size:
        raise ValueError("Il numero di client è maggiore del numero di esempi nel dataset.")
    
    if client_weights is not None:
        if len(client_weights) != num_clients:
            raise ValueError("La lunghezza di client_weights deve essere uguale a num_clients.")
        if not np.isclose(sum(client_weights), 1.0, atol=1e-4):
            raise ValueError("La somma dei pesi in client_weights deve essere 1.0.")
        sizes = [int(total_size * w) for w in client_weights]
        # Aggiustamento per compensare eventuali arrotondamenti
        diff = total_size - sum(sizes)
        sizes[-1] += diff
    else:
        # Distribuzione uniforme se non specificato
        sizes = [total_size // num_clients] * num_clients
        for i in range(total_size % num_clients):
            sizes[i] += 1

    os.makedirs(idx_root, exist_ok=True)

    # Indici iniziali
    indices = np.arange(total_size)
    if stratify:
        # Strategia: ripetuti StratifiedShuffleSplit progressivi per rispettare i pesi
        remaining_indices = indices.copy()
        current_X = X.copy()
        current_y = y.copy()

        for client_id in range(num_clients):
            client_size = sizes[client_id]

            if client_size >= len(current_X):
                client_indices = remaining_indices
            else:
                sss = StratifiedShuffleSplit(n_splits=1, train_size=client_size, random_state=42)
                train_idx, _ = next(sss.split(current_X, current_y))
                client_indices = remaining_indices[train_idx]

                mask = np.ones(len(current_X), dtype=bool)
                mask[train_idx] = False
                current_X = current_X[mask]
                current_y = current_y[mask]
                remaining_indices = remaining_indices[mask]

            # Salvataggio
            site_idx_file_name = os.path.join(idx_root, f'site-{client_id + 1}.npy')
            np.save(site_idx_file_name, client_indices)
            print(f"File {site_idx_file_name} creato con successo!")

            client_csv_file = os.path.join(idx_root, f'site-{client_id + 1}.csv')
            df.iloc[client_indices].to_csv(client_csv_file, index=False)
            print(f"File {client_csv_file} creato con successo!")
    else:
        np.random.shuffle(indices)
        start = 0
        for client_id in range(num_clients):
            end = start + sizes[client_id]
            client_indices = indices[start:end]
            site_idx_file_name = os.path.join(idx_root, f'site-{client_id + 1}.npy')
            np.save(site_idx_file_name, client_indices)
            print(f"File {site_idx_file_name} creato con successo!")

            client_csv_file = os.path.join(idx_root, f'site-{client_id + 1}.csv')
            df.iloc[client_indices].to_csv(client_csv_file, index=False)
            print(f"File {client_csv_file} creato con successo!")
            start = end

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    parser = argparse.ArgumentParser(description="Script per suddividere il dataset in file per ogni client.")
    parser.add_argument('num_clients', type=int, help="Numero di client.")
    parser.add_argument('--stratify', action='store_true', help="Usa stratificazione per le classi.")
    parser.add_argument('--weights', type=float, nargs='+', help="Percentuali di dati per ciascun client (devono sommare a 1.0)")

    args = parser.parse_args()

    data_path = '../dataset/data_train.csv'
    idx_root = '/tmp/mimic_data'

    os.makedirs(idx_root, exist_ok=True)
    shutil.copy(data_path, idx_root)

    create_site_files(data_path, idx_root, args.num_clients, stratify=args.stratify, client_weights=args.weights)

if __name__ == "__main__":
    main()
