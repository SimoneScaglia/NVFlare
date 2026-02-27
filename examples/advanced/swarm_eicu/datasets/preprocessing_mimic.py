import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def stratified_split(df, label_col, n_splits, train_total_size=None, train_ratio=0.7, random_state=42):
    if train_total_size:
        train_size_per_split = train_total_size // n_splits
    else:
        train_size_per_split = int(len(df) * train_ratio // n_splits)
        train_total_size = train_size_per_split * n_splits

    # Stratified sample for train
    df_train, _ = train_test_split(
        df, 
        train_size=train_total_size, 
        stratify=df[label_col], 
        random_state=random_state
    )
    # Split train in n_splits
    train_splits = []
    for i in range(n_splits):
        split = df_train.groupby(label_col, group_keys=False).apply(
            lambda x: x.sample(
                n=int(np.round(len(x) * train_size_per_split / len(df_train))),
                random_state=random_state + i
            )
        )
        train_splits.append(split.reset_index(drop=True))

    # For each train split, create a test split (stratified, 30% of train split size)
    test_splits = []
    for i, train_split in enumerate(train_splits):
        test_size = int(np.round(len(train_split) * (1 - train_ratio) / train_ratio))
        df_remaining = df.drop(train_split.index)
        test_split, _ = train_test_split(
            df_remaining, 
            train_size=test_size, 
            stratify=df_remaining[label_col], 
            random_state=random_state + i
        )
        test_splits.append(test_split.reset_index(drop=True))

    return train_splits, test_splits

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Split mimiciv.csv in 5 stratified train/test datasets.")
    parser.add_argument("input_csv", help="Input CSV file (e.g., mimiciv.csv)")
    parser.add_argument("--train_total_size", type=int, default=None, help="Total size of all train datasets (optional)")
    parser.add_argument("--label_col", type=str, default="sepsi", help="Name of the label column for stratification (default: sepsi)")
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)
    n_splits = 5

    train_splits, test_splits = stratified_split(
        df, 
        label_col=args.label_col, 
        n_splits=n_splits, 
        train_total_size=args.train_total_size
    )

    # Salva i file come richiesto: 1_train.csv, 1_test.csv, ...
    for i in range(n_splits):
        train_splits[i].to_csv(f"{i+1}_train.csv", index=False)
        test_splits[i].to_csv(f"{i+1}_test.csv", index=False)
        print(f"Creati {i+1}_train.csv ({len(train_splits[i])} righe) e {i+1}_test.csv ({len(test_splits[i])} righe)")

    # Crea un test set unico concatenando tutti i test set
    all_test = pd.concat(test_splits, ignore_index=True)
    all_test.to_csv("test.csv", index=False)
    print(f"Creato test.csv ({len(all_test)} righe)")