import pandas as pd
import numpy as np
import os
import sys

def main():
    if len(sys.argv) != 3:
        print("Uso: python prepare_data_heterogeneous.py <iterazione> <num_sites>")
        sys.exit(1)

    iterazione = int(sys.argv[1])
    num_sites = int(sys.argv[2])

    if num_sites < 2:
        print("num_sites deve essere almeno 2 (uno per B=1 e almeno uno per B=0).")
        sys.exit(1)

    # Percorso del file sorgente
    script_dir = os.path.dirname(os.path.abspath(__file__))
    source_file = os.path.join(script_dir, "..", "mimic", "dataset", "mimiciii_train.csv")
    source_file = os.path.abspath(source_file)

    # Caricamento CSV
    df = pd.read_csv(source_file)

    df_b1 = df[df['is_sepsis'] == 1]
    df_b0 = df[df['is_sepsis'] == 0]

    df_b0_shuffled = df_b0.sample(frac=1, random_state=iterazione).reset_index(drop=True)

    b0_splits = np.array_split(df_b0_shuffled, num_sites - 1)

    # Percorso di destinazione
    output_dir = "/tmp/mimic_data"
    os.makedirs(output_dir, exist_ok=True)

    df_b1.to_csv(os.path.join(output_dir, "site-1.csv"), index=False)

    for i, part in enumerate(b0_splits):
        dest_idx = i + 2
        part.to_csv(os.path.join(output_dir, f"site-{dest_idx}.csv"), index=False)

    print(f"File salvati in '{output_dir}' con {num_sites} siti totali.")

if __name__ == "__main__":
    main()
