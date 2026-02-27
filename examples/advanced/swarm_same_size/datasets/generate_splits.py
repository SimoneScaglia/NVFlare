import os
import pandas as pd
from sklearn.utils import shuffle

base_folder = "mimic_iv"

# Carica il dataset
df = pd.read_csv(f"{base_folder}/train.csv")

# Controlla la distribuzione iniziale
print("Distribuzione iniziale:")
print(df["is_sepsis"].value_counts(normalize=True))

# Parametri
n_files = 5
rows_per_file = 8000
rows_total = n_files * rows_per_file
output_dir = f"{base_folder}/same_size/{n_files}nodes"
os.makedirs(output_dir, exist_ok=True)

# Target per iterazione
n_pos_per_file = int(rows_per_file * 0.167)
n_neg_per_file = rows_per_file - n_pos_per_file
n_pos_total = n_pos_per_file * n_files
n_neg_total = n_neg_per_file * n_files

for iteration in range(10):
    print(f"\n--- Iterazione {iteration} ---")

    df_shuffled = shuffle(df, random_state=iteration).reset_index(drop=True)

    positives = df_shuffled[df_shuffled["is_sepsis"] == 1].head(n_pos_total).reset_index(drop=True)
    negatives = df_shuffled[df_shuffled["is_sepsis"] == 0].head(n_neg_total).reset_index(drop=True)

    positives = shuffle(positives, random_state=iteration).reset_index(drop=True)
    negatives = shuffle(negatives, random_state=iteration).reset_index(drop=True)

    pos_index = 0
    neg_index = 0

    for i in range(n_files):
        pos_chunk = positives.iloc[pos_index: pos_index + n_pos_per_file]
        neg_chunk = negatives.iloc[neg_index: neg_index + n_neg_per_file]
        pos_index += n_pos_per_file
        neg_index += n_neg_per_file

        chunk = pd.concat([pos_chunk, neg_chunk]).sample(frac=1, random_state=iteration+i).reset_index(drop=True)

        filename = os.path.join(output_dir, f"node{i+1}_{iteration}.csv")
        chunk.to_csv(filename, index=False)
        print(f"Creato {filename} con {len(chunk)} righe")

        dist = chunk["is_sepsis"].value_counts()
        print(f"Distribuzione {filename}: {dist.to_dict()}")
