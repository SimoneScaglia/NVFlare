"""
Questo script esegue il preprocessing del dataset eICU e crea train/test split
per diverse configurazioni di nodi con numero fisso di righe di train.

Configurazioni:
  - 5 nodi,  2500 righe train, test = 30% di (train+test)
  - 10 nodi, 1250 righe train, test = 30% di (train+test)
  - 20 nodi,  625 righe train, test = 30% di (train+test)
  - 25 nodi,  500 righe train, test = 30% di (train+test)

Per ogni configurazione vengono preprocessati TUTTI gli ospedali che hanno
abbastanza dati (train + test). La selezione casuale di N ospedali
verrà fatta a runtime.
L'imputazione viene fatta per ospedale: fit su train, transform su train e test.
"""

import math
import os

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

# Configurazioni: (n_nodi, righe_train)
CONFIGS = [
    (5, 2500),
    (10, 1250),
    (20, 625),
    (25, 500),
]

# File di input
datafile = "eicu_70k_rows.csv"
df = pd.read_csv(datafile)

# Verifica che tutte le colonne specificate siano presenti
expected_columns = [
    "hospitalid", "patientunitstayid", "uniquepid",
    "sepsis_explicit_hospital_acquired", "sepsis_angus_implicit_hospital_acquired",
    "sepsis_hospital_acquired", "sepsis_dx_atemporal", "vancomycin",
    "piperacillin", "acyclovir", "ciprofloxacin", "epinephrine",
    "norepinephrine", "vasopressin", "phenylephrine", "dopamine",
    "metoprolol", "kcl", "omeprazole", "pantoprazole", "ibp_sys_min_cat",
    "ibp_sys_mean_cat", "ibp_sys_max_cat", "ibp_dias_min_cat",
    "ibp_dias_mean_cat", "ibp_dias_max_cat", "nibp_sys_min_cat",
    "nibp_sys_mean_cat", "nibp_sys_max_cat", "nibp_dias_min_cat",
    "nibp_dias_mean_cat", "nibp_dias_max_cat", "hr_min_cat", "hr_mean_cat",
    "hr_max_cat", "rr_min_cat", "rr_mean_cat", "rr_max_cat", "spo2_min_cat",
    "spo2_mean_cat", "spo2_max_cat", "temp_min_cat", "temp_mean_cat",
    "temp_max_cat", "wbc_min_cat", "wbc_mean_cat", "wbc_max_cat"
]

missing_columns = [col for col in expected_columns if col not in df.columns]
if missing_columns:
    print(f"Attenzione: Colonne mancanti nel dataset: {missing_columns}")
else:
    print("Tutte le colonne attese sono presenti nel dataset")

# Rimuovi le colonne non necessarie
columns_to_remove = [
    "patientunitstayid", "uniquepid",
    "sepsis_explicit_hospital_acquired", "sepsis_angus_implicit_hospital_acquired",
    "sepsis_dx_atemporal"
]
df = df.drop(columns=columns_to_remove)
print(f"Colonne dopo la rimozione: {list(df.columns)}")

# Preprocessing: converte valori booleani e categorici
for column in df.columns:
    if df[column].dtype == 'bool':
        df[column] = df[column].astype(int)
    else:
        unique_vals = df[column].dropna().unique()
        if any(isinstance(val, str) for val in unique_vals):
            if any(val in ['low', 'medium', 'high'] for val in unique_vals if isinstance(val, str)):
                df[column] = df[column].replace({'low': 0, 'medium': 1, 'high': 2})

# Conta righe per ospedale e ordina in modo decrescente
hospital_counts = df['hospitalid'].value_counts().sort_values(ascending=False)
print(f"\nOspedali disponibili (ordinati per righe): {len(hospital_counts)}")
print(hospital_counts.head(30).to_string())

# Processa ogni configurazione
for n_nodes, train_rows in CONFIGS:
    # 70% train, 30% test => test = train * 3/7
    test_rows = math.ceil(train_rows * 3 / 7)
    total_rows_needed = train_rows + test_rows
    
    print(f"\n{'='*70}")
    print(f"Configurazione: {n_nodes} nodi, {train_rows} train, {test_rows} test, "
        f"{total_rows_needed} totale per ospedale")
    print(f"{'='*70}")
    
    # Crea la cartella di output
    output_dir = f"data_fixed_rows/{n_nodes}nodes"
    os.makedirs(output_dir, exist_ok=True)
    
    # Seleziona TUTTI gli ospedali con abbastanza righe
    eligible_hospitals = hospital_counts[hospital_counts >= total_rows_needed].index.tolist()
    
    if len(eligible_hospitals) < n_nodes:
        print(f"  ATTENZIONE: Solo {len(eligible_hospitals)} ospedali hanno >= {total_rows_needed} righe, "
            f"ne servono {n_nodes}. Configurazione saltata.")
        continue
    
    selected_hospitals = eligible_hospitals
    print(f"  Ospedali idonei: {len(selected_hospitals)} (minimi richiesti: {n_nodes})")
    print(f"  Ospedali: {[int(h) for h in selected_hospitals]}")
    
    for hospital_id in selected_hospitals:
        print(f"\n  Processing hospitalid: {int(hospital_id)}")
        
        # Filtra i dati per l'ospedale corrente
        hospital_df = df[df['hospitalid'] == hospital_id].copy()
        n_rows = len(hospital_df)
        print(f"    Righe disponibili: {n_rows}")
        
        # Tronca a total_rows_needed mantenendo la stratificazione
        if n_rows > total_rows_needed:
            X_temp = hospital_df.drop(columns=["sepsis_hospital_acquired", "hospitalid"])
            y_temp = hospital_df["sepsis_hospital_acquired"]
            
            X_temp_reduced, _, y_temp_reduced, _ = train_test_split(
                X_temp, y_temp, train_size=total_rows_needed, stratify=y_temp, random_state=42
            )
            
            hospital_df = pd.concat([X_temp_reduced, y_temp_reduced], axis=1)
            hospital_df["hospitalid"] = hospital_id
            print(f"    Troncato a {total_rows_needed} righe (rimosse {n_rows - total_rows_needed})")
        
        # Prepara target e features
        y = hospital_df["sepsis_hospital_acquired"]
        X = hospital_df.drop(columns=["sepsis_hospital_acquired", "hospitalid"])
        
        # Divisione train/test stratificata
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, stratify=y, train_size=train_rows, test_size=test_rows, random_state=42
        )
        
        # Imputazione valori mancanti (fit su train, transform su train e test)
        imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
        X_train = pd.DataFrame(imp_mean.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
        X_test = pd.DataFrame(imp_mean.transform(X_test), columns=X_test.columns, index=X_test.index)
        
        # Aggiungi la colonna target come ultima colonna
        X_train["sepsis_hospital_acquired"] = y_train
        X_test["sepsis_hospital_acquired"] = y_test
        
        # Salva i file
        train_filename = f"{output_dir}/{int(hospital_id)}_train.csv"
        test_filename = f"{output_dir}/{int(hospital_id)}_test.csv"
        
        X_train.to_csv(train_filename, index=False)
        X_test.to_csv(test_filename, index=False)
        
        print(f"    Creati: {train_filename} ({len(X_train)} righe)")
        print(f"            {test_filename} ({len(X_test)} righe)")

print(f"\nProcesso completato. File salvati nella cartella 'data_fixed_rows/'")