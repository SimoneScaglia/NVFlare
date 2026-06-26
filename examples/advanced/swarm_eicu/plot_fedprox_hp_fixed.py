import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# ==================================================
# CONFIGURAZIONE
# ==================================================
BASE_SWARM_DIR = Path("fedprox_new_results/hp_fixed")
BASE_CENTRAL_DIR = Path("new_results")
OUTPUT_DIR = Path("plots_results/mu_analysis/hp_fixed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Dataset e iperparametri
eicu_variants_hyperparamers = {
    "data_20k_entire":              [0.0005, 32],
    "data_20k_fixed_rows_entire":   [0.0001,  8],
    "data_20k_fixed_rows_separate": [0.0001,  8],
    "data_20k_separate":            [0.0005, 32],
    "data_entire":                  [0.0001,  8],
    "data_fixed_rows_entire":       [0.0001,  8],
    "data_fixed_rows_separate":     [0.0001,  8],
    "data_separate":                [0.0001,  8],
}

# Flag per includere il modello centrale
PLOT_CENTRAL = True   # cambia in False se vuoi solo i risultati swarm

MU_VALUES_TO_PLOT = None

MU_MATCH_ATOL = 1e-12
MU_MATCH_RTOL = 1e-6

# ==================================================
# FUNZIONI DI PARSING E CARICAMENTO (CENTRAL)
# ==================================================
def parse_lr_from_dir(dir_name):
    lr_part = dir_name.split('lr')[-1].split('_')[0]
    lr_str = lr_part.replace('-', '.')
    return float(lr_str)

def parse_bs_from_dir(dir_name):
    bs_part = dir_name.split('bs')[-1]
    return int(bs_part)

def parse_epochs_from_dir(dir_name):
    return int(dir_name.split('_')[0])

def parse_iteration_from_dir(dir_name):
    return int(dir_name.split('_')[1])

def compute_iteration_metric(df, metric):
    """Calcola media pesata (o semplice) di una metrica su un'iterazione."""
    if "splits" in df.columns:
        split_sum = df["splits"].sum()
        if abs(split_sum - 100) < 1e-6:
            return (df[metric] * df["splits"]).sum() / split_sum
        else:
            return df[metric].mean()
    else:
        return df[metric].mean()

def compute_separate_testset_iteration_metrics(df_epoch, metric):
    """Restituisce lista di metriche per ogni iterazione in un'epoca."""
    if "iteration" in df_epoch.columns:
        return [compute_iteration_metric(df_iter, metric) for _, df_iter in df_epoch.groupby("iteration")]
    return [compute_iteration_metric(df_epoch, metric)]

def is_separate_testset_epoch(df_epoch):
    """Determina se l'epoca contiene più utenti (testset separati)."""
    if "user" not in df_epoch.columns:
        return False
    users = {str(u).strip().lower() for u in df_epoch["user"].dropna().astype(str)}
    if not users:
        return False
    if "swarm_global" in users:
        return False
    if users == {"central"}:
        return False
    return len(users) > 1

def get_existing_csv_path(dir_path, csv_filenames):
    """Cerca il primo file CSV esistente tra quelli forniti."""
    if isinstance(csv_filenames, str):
        csv_filenames = [csv_filenames]
    for filename in csv_filenames:
        csv_path = dir_path / filename
        if csv_path.exists():
            return csv_path
    return None

def normalize_mu_values(mu_values):
    """Converte valori mu (float/string) in float, supportando formati diversi."""
    normalized = []
    for mu in mu_values:
        if mu is None:
            continue
        if isinstance(mu, str):
            mu = mu.strip().replace(",", ".")
            if not mu:
                continue
        try:
            normalized.append(float(mu))
        except (TypeError, ValueError):
            continue
    return normalized

def load_central_results(base_dir, dataset_name, csv_filenames):
    """
    Carica i risultati centrali (senza mu) da new_results/<dataset_name>/...
    Restituisce due dizionari annidati: auc_results[lr][bs][epoch] = lista,
                                loss_results[lr][bs][epoch] = lista
    """
    base_path = Path(base_dir) / dataset_name
    if not base_path.exists():
        return {}, {}

    auc_results = {}
    loss_results = {}

    for dir_path in base_path.iterdir():
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
                    iter_aucs = compute_separate_testset_iteration_metrics(df_epoch, 'auc')
                    iter_losses = compute_separate_testset_iteration_metrics(df_epoch, 'loss') if 'loss' in df_epoch.columns else []
                else:
                    iter_aucs = [compute_iteration_metric(df_epoch, 'auc')]
                    iter_losses = [compute_iteration_metric(df_epoch, 'loss')] if 'loss' in df_epoch.columns else []

                # Salva AUC
                auc_results.setdefault(lr, {}).setdefault(bs, {}).setdefault(epoch_int, []).extend(iter_aucs)
                # Salva loss
                if iter_losses:
                    loss_results.setdefault(lr, {}).setdefault(bs, {}).setdefault(epoch_int, []).extend(iter_losses)
        except Exception as e:
            print(f"Errore nel processare {csv_path}: {e}")

    return auc_results, loss_results

# ==================================================
# CICLO PRINCIPALE
# ==================================================
for key, (lr, bs) in eicu_variants_hyperparamers.items():
    print(f"\nProcesso dataset: {key} (lr={lr}, bs={bs})")

    # --- 1. Carica risultati swarm ---
    swarm_dir = BASE_SWARM_DIR / f"eicu_{key}_testset" / f"150_lr{lr:.5f}_bs{bs}"
    # Decidi quale CSV usare in base alla presenza di "_entire" nel nome
    if "_entire" in key:
        swarm_csv_name = "swarm_results_entire_testset.csv"
    else:
        swarm_csv_name = "swarm_results.csv"

    swarm_csv_path = swarm_dir / swarm_csv_name
    if not swarm_csv_path.exists():
        print(f"  File swarm non trovato: {swarm_csv_path}. Saltato.")
        continue

    try:
        df_swarm = pd.read_csv(swarm_csv_path)
        # Controlla che esistano le colonne necessarie
        required_cols = {'epoch', 'fedprox_mu', 'auc', 'loss', 'iteration'}
        missing = required_cols - set(df_swarm.columns)
        if missing:
            print(f"  Colonne mancanti nel CSV swarm: {missing}. Saltato.")
            continue

        # Converti fedprox_mu in float (gestisce notazioni diverse)
        df_swarm['fedprox_mu'] = pd.to_numeric(df_swarm['fedprox_mu'], errors='coerce')
        df_swarm.dropna(subset=['fedprox_mu'], inplace=True)

        # Raggruppa per (mu, epoch) e calcola la media su iterazioni
        swarm_agg = df_swarm.groupby(['fedprox_mu', 'epoch']).agg(
            auc_mean=('auc', 'mean'),
            loss_mean=('loss', 'mean')
        ).reset_index()

        # Ordina per sicurezza
        swarm_agg.sort_values(['fedprox_mu', 'epoch'], inplace=True)

    except Exception as e:
        print(f"  Errore nella lettura/aggregazione swarm: {e}")
        continue

    # --- 2. Carica risultati centrali (se PLOT_CENTRAL) ---
    central_auc = {}
    central_loss = {}
    if PLOT_CENTRAL:
        auc_res, loss_res = load_central_results(BASE_CENTRAL_DIR, f"eicu_{key}_testset", "central_results.csv")
        # Estrai i dati corrispondenti a lr e bs correnti
        if lr in auc_res and bs in auc_res[lr]:
            # Per ogni epoca, calcola la media delle iterazioni
            central_auc = {ep: np.mean(vals) for ep, vals in auc_res[lr][bs].items()}
        else:
            print(f"  Nessun centrale per lr={lr}, bs={bs}")

        if lr in loss_res and bs in loss_res[lr]:
            central_loss = {ep: np.mean(vals) for ep, vals in loss_res[lr][bs].items()}
        # Se non ci sono loss, semplicemente non plottiamo la loss centrale

    # --- 3. Creazione del plot ---
    # Lista dei valori unici di mu, ordinati dal più piccolo al più grande
    mu_values = sorted(swarm_agg['fedprox_mu'].unique())

    # Se richiesto, filtra i mu da plottare con match robusto rispetto ai valori del CSV
    if MU_VALUES_TO_PLOT is not None:
        requested_mu = normalize_mu_values(MU_VALUES_TO_PLOT)
        matched_mu = [
            mu
            for mu in mu_values
            if any(np.isclose(mu, req_mu, atol=MU_MATCH_ATOL, rtol=MU_MATCH_RTOL) for req_mu in requested_mu)
        ]
        missing_mu = [
            req_mu
            for req_mu in requested_mu
            if not any(np.isclose(req_mu, mu, atol=MU_MATCH_ATOL, rtol=MU_MATCH_RTOL) for mu in mu_values)
        ]

        if missing_mu:
            print(f"  Mu richiesti non trovati nel CSV: {sorted(set(missing_mu))}")

        if not matched_mu:
            print("  Nessun mu selezionato disponibile per questo dataset. Saltato.")
            continue

        mu_values = matched_mu
        swarm_agg = swarm_agg[swarm_agg['fedprox_mu'].isin(mu_values)]

    # Per avere colori distinti, prendiamo una mappa colori
    cmap = plt.get_cmap('tab10')
    colors = [cmap(i % 10) for i in range(len(mu_values))]

    fig, (ax_auc, ax_loss) = plt.subplots(2, 1, figsize=(20, 10), sharex=True)

    # --- 3a. Subplot AUC ---
    for idx, mu in enumerate(mu_values):
        sub = swarm_agg[swarm_agg['fedprox_mu'] == mu]
        ax_auc.plot(sub['epoch'], sub['auc_mean'],
                    color=colors[idx], marker='o', linestyle='-',
                    label=f'μ={mu}')
    if central_auc:
        # Aggiungi linea centrale
        ep_central = sorted(central_auc.keys())
        auc_central_vals = [central_auc[ep] for ep in ep_central]
        ax_auc.plot(ep_central, auc_central_vals,
                    color='black', linestyle='--', linewidth=2,
                    marker='s', label='Central')
    ax_auc.set_ylabel('AUC')
    ax_auc.set_ylim(0.5, 0.9)
    ax_auc.set_title(f'Dataset: eicu_{key}  (lr={lr}, bs={bs})', fontsize=12)
    ax_auc.legend(title='FedProx μ', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax_auc.grid(True, alpha=0.3)

    # --- 3b. Subplot Loss ---
    for idx, mu in enumerate(mu_values):
        sub = swarm_agg[swarm_agg['fedprox_mu'] == mu]
        ax_loss.plot(sub['epoch'], sub['loss_mean'],
                    color=colors[idx], marker='o', linestyle='-',
                    label=f'μ={mu}')
    if central_loss:
        ep_central = sorted(central_loss.keys())
        loss_central_vals = [central_loss[ep] for ep in ep_central]
        ax_loss.plot(ep_central, loss_central_vals,
                    color='black', linestyle='--', linewidth=2,
                    marker='s', label='Central')
    ax_loss.set_xlabel('Epoch')
    ax_loss.set_ylabel('Loss')
    ax_loss.set_ylim(0.3, 0.7)
    ax_loss.legend(title='FedProx μ', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax_loss.grid(True, alpha=0.3)

    plt.tight_layout()
    # Salva
    output_path = OUTPUT_DIR / f"eicu_{key}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Plot salvato: {output_path}")

print("\nTutti i plot sono stati generati.")
