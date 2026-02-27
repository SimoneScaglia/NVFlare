import pandas as pd
from sklearn.model_selection import train_test_split
import argparse

def split_csv_stratified(input_csv, train_csv='newexp_train.csv', test_csv='newexp_test.csv', train_size=10000, random_state=42):
    # Legge il CSV
    df = pd.read_csv(input_csv)
    
    # Identifica la colonna target (ultima colonna)
    target_col = df.columns[-1]
    y = df[target_col]

    # Controlla che ci siano abbastanza righe
    if len(df) <= train_size:
        raise ValueError(f"Il dataset ha solo {len(df)} righe, servono almeno {train_size + 1}.")
    
    # Calcola la proporzione per ottenere ~10k righe nel train
    train_ratio = train_size / len(df)

    # Split stratificato
    train_df, test_df = train_test_split(
        df,
        stratify=y,
        train_size=train_ratio,
        random_state=random_state,
        shuffle=True
    )

    # Salva i CSV
    train_df.to_csv(train_csv, index=False)
    test_df.to_csv(test_csv, index=False)

    print(f"Creati i file:")
    print(f"- {train_csv}: {len(train_df)} righe")
    print(f"- {test_csv}: {len(test_df)} righe")
    print(f"Stratificato sulla colonna: {target_col}")

if __name__ == "__main__":
    split_csv_stratified("mimiciv.csv")
