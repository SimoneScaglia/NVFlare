#!/usr/bin/env python3
"""
compute_robustness_index_combined.py

Calcola il Robustness Index (RI) combinato per ogni coppia (lr, bs)
considerando sia i risultati Swarm che Central.
La performance tiene conto della distanza dal massimo raggiunto in ciascun setting.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import argparse
from collections import defaultdict

# ------------------------------------------------------------------------------
# Helper functions (adattate dallo script originale)
# ------------------------------------------------------------------------------

def parse_lr_from_dir(dir_name):
    """Extract learning rate from directory name (format: *_lrX_bsY)."""
    lr_part = dir_name.split('lr')[-1].split('_')[0]
    lr_str = lr_part.replace('-', '.')
    return float(lr_str)

def parse_bs_from_dir(dir_name):
    """Extract batch size from directory name (format: *_bsY)."""
    bs_part = dir_name.split('bs')[-1]
    return int(bs_part)

def parse_epochs_from_dir(dir_name):
    """Extract total epochs from directory name (first number before first underscore)."""
    return int(dir_name.split('_')[0])

def parse_iteration_from_dir(dir_name):
    """Extract iteration number from directory name (second number)."""
    return int(dir_name.split('_')[1])

def compute_iteration_metrics(df):
    """
    Given a DataFrame for one iteration (possibly with multiple splits),
    compute the average AUC and Loss for that iteration.
    If 'splits' column exists and sum(splits)==100, use weighted average,
    otherwise simple mean.
    Returns (avg_auc, avg_loss).
    """
    if "splits" in df.columns:
        split_sum = df["splits"].sum()
        if split_sum == 100:
            w = df["splits"] / split_sum
            avg_auc = (df["auc"] * w).sum()
            avg_loss = (df["loss"] * w).sum() if "loss" in df.columns else np.nan
        else:
            avg_auc = df["auc"].mean()
            avg_loss = df["loss"].mean() if "loss" in df.columns else np.nan
    else:
        avg_auc = df["auc"].mean()
        avg_loss = df["loss"].mean() if "loss" in df.columns else np.nan
    return avg_auc, avg_loss


def compute_separate_testset_iteration_metrics(df_epoch):
    """
    Compute one (auc, loss) per iteration for separate-testset evaluations.
    """
    if "iteration" in df_epoch.columns:
        metrics = []
        for _, df_iter in df_epoch.groupby("iteration"):
            avg_auc, avg_loss = compute_iteration_metrics(df_iter)
            metrics.append((avg_auc, avg_loss))
        return metrics
    avg_auc, avg_loss = compute_iteration_metrics(df_epoch)
    return [(avg_auc, avg_loss)]


def is_separate_testset_epoch(df_epoch):
    """
    Detect separate-testset rows by checking whether we have per-user rows.
    """
    if "user" not in df_epoch.columns:
        return False
    users = {str(u).strip().lower() for u in df_epoch["user"].dropna().astype(str)}
    if not users:
        return False
    if "swarm_global" in users or users == {"central"}:
        return False
    return len(users) > 1


def get_existing_csv_path(dir_path, csv_filenames):
    """Return the first existing CSV path from a filename or list of filenames."""
    if isinstance(csv_filenames, str):
        csv_filenames = [csv_filenames]
    for filename in csv_filenames:
        csv_path = dir_path / filename
        if csv_path.exists():
            return csv_path
    return None


def load_results_for_config(base_dir, subdir, csv_filenames):
    """
    Load all results for a given configuration type (swarm or central).
    Returns a nested dictionary:
        results[lr][bs][epoch] = (list_of_auc, list_of_loss)
    """
    results = {}
    dataset_path = Path(base_dir) / subdir
    if not dataset_path.exists():
        print(f"Warning: path does not exist: {dataset_path}")
        return results

    for dir_path in dataset_path.iterdir():
        if not dir_path.is_dir():
            continue
        dir_name = dir_path.name
        try:
            _ = parse_iteration_from_dir(dir_name)
            lr = parse_lr_from_dir(dir_name)
            bs = parse_bs_from_dir(dir_name)
        except Exception:
            continue

        csv_path = get_existing_csv_path(dir_path, csv_filenames)
        if csv_path is None:
            continue

        try:
            df = pd.read_csv(csv_path)
            if 'epoch' not in df.columns:
                continue

            for epoch_value, df_epoch in df.groupby('epoch'):
                epoch_int = int(epoch_value)
                if is_separate_testset_epoch(df_epoch):
                    iter_metrics = compute_separate_testset_iteration_metrics(df_epoch)
                else:
                    avg_auc, avg_loss = compute_iteration_metrics(df_epoch)
                    iter_metrics = [(avg_auc, avg_loss)]

                auc_list = [m[0] for m in iter_metrics if not np.isnan(m[0])]
                loss_list = [m[1] for m in iter_metrics if not np.isnan(m[1])]

                if auc_list or loss_list:
                    results.setdefault(lr, {}).setdefault(bs, {}).setdefault(epoch_int, (auc_list, loss_list))

        except Exception as e:
            print(f"Error processing {csv_path}: {e}")

    return results


# ------------------------------------------------------------------------------
# Funzioni per il calcolo del Robustness Index combinato
# ------------------------------------------------------------------------------

def average_epoch_metrics(results):
    """
    Converte il dizionario annidato in:
        configs: list of (lr, bs)
        epoch_data: dict mapping (lr, bs) -> dict {epoch: (mean_auc, mean_loss)}
    """
    configs = []
    epoch_data = {}
    for lr, bs_dict in results.items():
        for bs, epochs in bs_dict.items():
            config = (lr, bs)
            configs.append(config)
            epoch_data[config] = {}
            for epoch, (auc_list, loss_list) in epochs.items():
                mean_auc = np.mean(auc_list) if auc_list else np.nan
                mean_loss = np.mean(loss_list) if loss_list else np.nan
                if not np.isnan(mean_auc) and not np.isnan(mean_loss):
                    epoch_data[config][epoch] = (mean_auc, mean_loss)
    return configs, epoch_data


def compute_setting_metrics(epoch_data):
    """
    Per un dato setting (swarm o central), calcola per ogni configurazione:
        - avg_auc_lastK, avg_loss_lastK
        - vol_auc, vol_loss
    Restituisce un DataFrame con colonne: config, avg_auc_lastK, avg_loss_lastK, vol_auc, vol_loss
    """
    rows = []
    for config, epochs in epoch_data.items():
        if not epochs:
            continue
        sorted_epochs = sorted(epochs.keys())
        auc_series = [epochs[e][0] for e in sorted_epochs]
        loss_series = [epochs[e][1] for e in sorted_epochs]

        K = len(sorted_epochs) # taking all epochs into account for average
        if K == 0:
            continue
        avg_auc_lastK = np.mean(auc_series[-K:])
        avg_loss_lastK = np.mean(loss_series[-K:])

        diff_auc = [abs(auc_series[i] - auc_series[i-1]) for i in range(1, len(auc_series))]
        diff_loss = [abs(loss_series[i] - loss_series[i-1]) for i in range(1, len(loss_series))]
        vol_auc = np.mean(diff_auc) if diff_auc else 0.0
        vol_loss = np.mean(diff_loss) if diff_loss else 0.0

        rows.append((config, avg_auc_lastK, avg_loss_lastK, vol_auc, vol_loss))

    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows, columns=['config', 'avg_auc_lastK', 'avg_loss_lastK', 'vol_auc', 'vol_loss'])
    return df


def compute_robustness_index_combined(swarm_results, central_results):
    """
    Calcola il Robustness Index combinato per ogni configurazione (lr, bs)
    a partire dai dizionari di risultati di swarm e central.

    Restituisce un DataFrame con colonne:
        lr, bs, robustness_index, performance_score, stability_score,
        (altre colonne di dettaglio)
    """
    # Estrai le serie temporali per ciascun setting
    _, swarm_epoch = average_epoch_metrics(swarm_results)
    _, central_epoch = average_epoch_metrics(central_results)

    # Uniamo le configurazioni presenti in entrambi (o in almeno uno?)
    # Per il calcolo combinato, vogliamo configurazioni che appaiono in entrambi.
    common_configs = set(swarm_epoch.keys()) & set(central_epoch.keys())
    if not common_configs:
        return pd.DataFrame()

    # Per ogni setting, calcoliamo le metriche grezze per tutte le configs (anche quelle non comuni, per la normalizzazione)
    df_swarm = compute_setting_metrics(swarm_epoch)
    df_central = compute_setting_metrics(central_epoch)

    if df_swarm.empty or df_central.empty:
        return pd.DataFrame()

    # Filtriamo solo le configurazioni comuni
    df_swarm = df_swarm[df_swarm['config'].isin(common_configs)]
    df_central = df_central[df_central['config'].isin(common_configs)]

    # Uniamo i due DataFrame
    df_swarm = df_swarm.rename(columns={
        'avg_auc_lastK': 'auc_swarm',
        'avg_loss_lastK': 'loss_swarm',
        'vol_auc': 'vol_auc_swarm',
        'vol_loss': 'vol_loss_swarm'
    })
    df_central = df_central.rename(columns={
        'avg_auc_lastK': 'auc_central',
        'avg_loss_lastK': 'loss_central',
        'vol_auc': 'vol_auc_central',
        'vol_loss': 'vol_loss_central'
    })
    df = pd.merge(df_swarm, df_central, on='config', suffixes=('', ''))

    # Calcoliamo i massimi e minimi per setting
    # AUC: massimi
    max_auc_swarm = df['auc_swarm'].max()
    max_auc_central = df['auc_central'].max()
    # Loss: minimi
    min_loss_swarm = df['loss_swarm'].min()
    min_loss_central = df['loss_central'].min()
    # Volatilità: min e max per normalizzazione
    min_vol_auc_swarm = df['vol_auc_swarm'].min()
    max_vol_auc_swarm = df['vol_auc_swarm'].max()
    min_vol_loss_swarm = df['vol_loss_swarm'].min()
    max_vol_loss_swarm = df['vol_loss_swarm'].max()
    min_vol_auc_central = df['vol_auc_central'].min()
    max_vol_auc_central = df['vol_auc_central'].max()
    min_vol_loss_central = df['vol_loss_central'].min()
    max_vol_loss_central = df['vol_loss_central'].max()

    def safe_norm(val, minv, maxv):
        if maxv - minv < 1e-12:
            return 0.5
        return (val - minv) / (maxv - minv)

    # Vicinanza al massimo per AUC (per setting)
    df['vicinanza_auc_swarm'] = df['auc_swarm'] / max_auc_swarm if max_auc_swarm > 0 else 0.5
    df['vicinanza_auc_central'] = df['auc_central'] / max_auc_central if max_auc_central > 0 else 0.5

    # Vicinanza alla loss minima (invertita)
    denom_loss_swarm = max(df['loss_swarm']) - min_loss_swarm
    denom_loss_central = max(df['loss_central']) - min_loss_central
    df['vicinanza_loss_swarm'] = df['loss_swarm'].apply(
        lambda x: 1 - safe_norm(x, min_loss_swarm, max(df['loss_swarm'])) if denom_loss_swarm > 0 else 0.5
    )
    df['vicinanza_loss_central'] = df['loss_central'].apply(
        lambda x: 1 - safe_norm(x, min_loss_central, max(df['loss_central'])) if denom_loss_central > 0 else 0.5
    )

    # Performance per setting (media delle vicinanze)
    df['perf_swarm'] = 0.5 * (df['vicinanza_auc_swarm'] + df['vicinanza_loss_swarm'])
    df['perf_central'] = 0.5 * (df['vicinanza_auc_central'] + df['vicinanza_loss_central'])

    # Stabilità per setting: normalizzazione della volatilità (1 - norm)
    df['stability_swarm'] = 0.5 * (
        (1 - safe_norm(df['vol_auc_swarm'], min_vol_auc_swarm, max_vol_auc_swarm)) +
        (1 - safe_norm(df['vol_loss_swarm'], min_vol_loss_swarm, max_vol_loss_swarm))
    )
    df['stability_central'] = 0.5 * (
        (1 - safe_norm(df['vol_auc_central'], min_vol_auc_central, max_vol_auc_central)) +
        (1 - safe_norm(df['vol_loss_central'], min_vol_loss_central, max_vol_loss_central))
    )

    # Combinazione con media geometrica
    df['performance_score'] = np.sqrt(df['perf_swarm'] * df['perf_central'])
    df['stability_score'] = np.sqrt(df['stability_swarm'] * df['stability_central'])

    # Robustness Index combinato
    df['robustness_index'] = 0.5 * df['performance_score'] + 0.5 * df['stability_score']

    # Estrai lr e bs
    df[['lr', 'bs']] = pd.DataFrame(df['config'].tolist(), index=df.index)

    # Ordina per RI decrescente
    df = df.sort_values('robustness_index', ascending=False).reset_index(drop=True)

    # Colonne di output richieste
    output_cols = ['lr', 'bs', 'robustness_index', 'performance_score', 'stability_score',
                   'auc_swarm', 'auc_central', 'loss_swarm', 'loss_central',
                   'vol_auc_swarm', 'vol_auc_central', 'vol_loss_swarm', 'vol_loss_central']
    df = df[output_cols]

    return df


# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Calcola il Robustness Index combinato (Swarm + Central) per ogni coppia (lr, bs)."
    )
    parser.add_argument(
        "--swarm-subdir",
        type=str,
        default=None,
        help="Swarm results subdirectory under new_results/",
    )
    parser.add_argument(
        "--central-subdir",
        type=str,
        default=None,
        help="Central results subdirectory under new_results/",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default=None,
        help="Output dataset label when using custom subdirs.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="robustness_results_combined",
        help="Directory where CSV output files will be saved.",
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    SWARM_BASE = "new_results"
    CENTRAL_BASE = "new_results"
    SWARM_CSV = ["swarm_results_entire_testset.csv", "swarm_results.csv"]
    CENTRAL_CSV = ["central_results.csv"]

    # Dataset mapping (come nel tuo script originale)
    datasets = {
        "mimiciii_total": "mimiciii_total_entire_testset_epochs",
        "mimiciv_total": "mimiciv_total_entire_testset_epochs",
        "mimiciv_fixed": "mimiciv_fixed_12500_entire_testset_epochs",
        "eicu_data_20k_entire_testset": "eicu_data_20k_entire_testset",
        "eicu_data_20k_separate_testset": "eicu_data_20k_separate_testset",
        "eicu_data_20k_fixed_rows_entire_testset": "eicu_data_20k_fixed_rows_entire_testset",
        "eicu_data_20k_fixed_rows_separate_testset": "eicu_data_20k_fixed_rows_separate_testset",
        "eicu_data_entire_testset": "eicu_data_entire_testset",
        "eicu_data_separate_testset": "eicu_data_separate_testset",
        "eicu_data_fixed_rows_entire_testset": "eicu_data_fixed_rows_entire_testset",
        "eicu_data_fixed_rows_separate_testset": "eicu_data_fixed_rows_separate_testset"
    }

    if args.swarm_subdir and args.central_subdir:
        custom_name = args.dataset_name or f"swarm_{args.swarm_subdir}__central_{args.central_subdir}"
        dataset_jobs = [(custom_name, args.swarm_subdir, args.central_subdir)]
    else:
        dataset_jobs = []
        for name, subdir in datasets.items():
            dataset_jobs.append((name, subdir, subdir))

    for dataset_name, swarm_subdir, central_subdir in dataset_jobs:
        print(f"\nProcessing {dataset_name}...")
        swarm_results = load_results_for_config(SWARM_BASE, swarm_subdir, SWARM_CSV)
        central_results = load_results_for_config(CENTRAL_BASE, central_subdir, CENTRAL_CSV)

        if not swarm_results or not central_results:
            print(f"  Dati mancanti per {dataset_name}. Saltato.")
            continue

        df_ri = compute_robustness_index_combined(swarm_results, central_results)

        if df_ri.empty:
            print(f"  Nessuna configurazione comune per {dataset_name}. Saltato.")
            continue

        best = df_ri.iloc[0]
        print(f"  Migliore configurazione per {dataset_name}:")
        print(f"    lr = {best['lr']:.6f}, bs = {best['bs']}")
        print(f"    Robustness Index = {best['robustness_index']:.2f}")
        print(f"    Performance Score = {best['performance_score']:.4f}")
        print(f"    Stability Score   = {best['stability_score']:.4f}")
        print("  ---")

if __name__ == "__main__":
    main()

# robustness_index = k * performance_score + (1 - k) * stability_score
# 
# dove
# 
# performance_score = radice_quadrata(performance_swarm * performance_central)
# 
# performance_swarm = (auc_proximity_swarm + loss_proximity_swarm) / 2
# performance_central = (auc_proximity_central + loss_proximity_central) / 2
# 
# auc_proximity_swarm = auc_swarm / massimo_auc_swarm
# auc_proximity_central = auc_central / massimo_auc_central
# loss_proximity_swarm = 1 - ((loss_swarm - minimo_loss_swarm) / (massimo_loss_swarm - minimo_loss_swarm))
# loss_proximity_central = 1 - ((loss_central - minimo_loss_central) / (massimo_loss_central - minimo_loss_central))
# 
# e
# 
# stability_score = radice_quadrata(stability_swarm * stability_central)
# 
# stability_swarm = ((1 - normalizza(volatility_auc_swarm)) + (1 - normalizza(volatility_loss_swarm))) / 2
# stability_central = ((1 - normalizza(volatility_auc_central)) + (1 - normalizza(volatility_loss_central))) / 2
# 
# normalizza(valore) = (valore - valore_minimo) / (valore_massimo - valore_minimo)
# 
# volatility_auc = media(valore_assoluto(auc_epoca_i - auc_epoca_i_meno_1))
# volatility_loss = media(valore_assoluto(loss_epoca_i - loss_epoca_i_meno_1))