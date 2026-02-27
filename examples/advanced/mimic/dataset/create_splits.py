import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold


def split_data(X: pd.DataFrame, y: pd.Series, num_nodes: int, user: int, iteration: int, split_frac: float):
    """
    Splits X and y into num_nodes+1 partitions: first partition of size split_frac,
    remaining data equally split into num_nodes parts via stratified k-fold.
    Returns the partition corresponding to 'user' (1-based index).
    """
    if num_nodes == 1:
        x1, x2, y1, y2 = train_test_split(
            X, y, stratify=y, train_size=split_frac,
            shuffle=True, random_state=iteration
        )
        if user == 1:
            return x1, y1
        elif user == 2:
            return x2, y2
        else:
            raise ValueError(f"User {user} out of range for num_nodes=1")

    x1, xr, y1, yr = train_test_split(
        X, y, stratify=y, train_size=split_frac,
        shuffle=True, random_state=iteration
    )
    skf = StratifiedKFold(n_splits=num_nodes, shuffle=True, random_state=iteration)
    features = list(X.columns)

    folds = []
    for _, test_idx in skf.split(xr, yr):
        x_fold = pd.DataFrame(np.array(xr)[test_idx], columns=features)
        y_fold = np.array(yr)[test_idx]
        folds.append((x_fold, y_fold))

    if user == 1:
        return x1, y1
    elif 2 <= user <= num_nodes + 1:
        return folds[user - 2]
    else:
        raise ValueError(f"User {user} out of valid range 1..{num_nodes+1}")


if __name__ == "__main__":
    datafile = "mimiciv_train.csv"
    experiment_name = "new_splits_mimiciv"

    # Nodes and their corresponding splits
    nodes_list = [9, 14, 19, 24]
    splits_exp = [
        # 9 nodes
        [(p, *[round((100-p)/9, 2)]*9) for p in [2.5,5,10,20,30,40,50,60,70,80,90]],
        # 14 nodes
        [(p, *[round((100-p)/14, 2)]*14) for p in [2.5,5,10,20,30,40,50,60,70,80,90]],
        # 19 nodes
        [(p, *[round((100-p)/19, 2)]*19) for p in [2.5,5,10,20,30,40,50,60,70,80,90]],
        # 24 nodes
        [(p, *[round((100-p)/24, 2)]*24) for p in [2.5,5,10,20,30,40,50,60,70,80,90]],
    ]

    for num_nodes, splits in zip(nodes_list, splits_exp):
        out_dir = f"{experiment_name}_{num_nodes}_nodes"
        os.makedirs(out_dir, exist_ok=True)

        for split in splits:
            split_frac = split[0] / 100.0
            labels = split[1:]

            for iteration in range(10):
                df = pd.read_csv(datafile)
                y_full = df.pop("is_sepsis")
                X_full = df.copy()

                # Users 2..N+1
                for user, label in enumerate(labels, start=2):
                    X_part, y_part = split_data(
                        X_full, y_full, num_nodes, user,
                        iteration, split_frac
                    )
                    X_part["is_sepsis"] = y_part
                    filename = f"{out_dir}/{user}_{label}_{iteration}.csv"
                    X_part.to_csv(filename, index=False)

                # User 1
                X1, y1 = split_data(
                    X_full, y_full, num_nodes, 1,
                    iteration, split_frac
                )
                X1["is_sepsis"] = y1
                filename1 = f"{out_dir}/1_{split[0]}_{iteration}.csv"
                X1.to_csv(filename1, index=False)
