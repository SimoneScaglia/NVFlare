import numpy as np
import pandas as pd
import os
import shutil
import argparse

def create_site_files(data_path, idx_root, num_clients, train_percentage=0.8):
    # Carica il dataset
    df = pd.read_csv(data_path)  # Sostituisci con il percorso giusto
    X = df.iloc[:, :-1].values  # Carica le feature
    y = df.iloc[:, -1].values  # Carica le etichette

    # Calcola la dimensione totale del dataset
    total_size = len(X)

    # Calcola la dimensione del training set per ogni client
    train_size_per_client = int((train_percentage * total_size)/num_clients)

    # Assicurati che la cartella di destinazione esista
    if not os.path.exists(idx_root):
        os.makedirs(idx_root)

    # Calcola il numero di esempi per ogni client (gestione divisione diseguale)
    indices = np.arange(total_size)
    np.random.shuffle(indices)  # Mescola gli indici in modo casuale

    # Se il numero di clienti è maggiore del numero di esempi, restituisci un errore
    if num_clients > total_size:
        raise ValueError("Il numero di client è maggiore del numero di esempi nel dataset.")

    # Dividi i dati tra i client
    start_idx = 0
    for client_id in range(1, num_clients + 1):
        # Determina la dimensione del training set per il client corrente
        # Se non siamo all'ultimo client, assegniamo `train_size_per_client` esempi
        if client_id < num_clients:
            end_idx = start_idx + train_size_per_client
        else:
            # Per l'ultimo client, assegniamo i rimanenti dati
            end_idx = total_size

        client_indices = indices[start_idx:end_idx]

        # Salva gli indici in un file .npy
        site_idx_file_name = os.path.join(idx_root, f'site-{client_id}.npy')
        np.save(site_idx_file_name, client_indices)

        # Aggiorna l'indice di partenza per il prossimo client
        start_idx = end_idx

        print(f"File {site_idx_file_name} creato con successo!")

def main():
    # Change working directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    # Impostazione del parser per la riga di comando
    parser = argparse.ArgumentParser(description="Script per suddividere il dataset in file per ogni client.")
    parser.add_argument('num_clients', type=int, help="Numero di client.")
    
    # Parse degli argomenti
    args = parser.parse_args()

    # Parametri fissi
    data_path = '../dataset/data.csv'  # Percorso del file CSV
    idx_root = '/tmp/mimic_data'  # Percorso dove salvare i file .npy
    train_percentage = 0.8  # Percentuale di dati da usare per il training (80% nel nostro esempio)

    # Crea la directory di destinazione se non esiste
    os.makedirs(idx_root, exist_ok=True)

    # Copia il file CSV nella cartella di destinazione
    shutil.copy(data_path, idx_root)

    # Chiamata alla funzione per creare i file per ogni client
    create_site_files(data_path, idx_root, args.num_clients, train_percentage)

if __name__ == "__main__":
    main()
