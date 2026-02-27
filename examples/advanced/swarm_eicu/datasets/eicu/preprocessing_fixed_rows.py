"""
Questo script esegue il preprocessing e divide il dataset in train e test per i 5 hospitalid con più dati.
Riduce il numero di righe per ogni ospedale a 3500 (se ne ha di più) mantenendo la stratificazione,
poi divide in 2500 train e 1000 test mantenendo la stratificazione.
"""

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
import os

# Crea la cartella data se non esiste
os.makedirs("data_fixed_rows", exist_ok=True)

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

# Verifica colonne mancanti
missing_columns = [col for col in expected_columns if col not in df.columns]
if missing_columns:
    print(f"Attenzione: Colonne mancanti nel dataset: {missing_columns}")
else:
    print("Tutte le colonne attese sono presenti nel dataset")

# PRIMO PASSO: Rimuovi le colonne specificate (tranne hospitalid e sepsis_hospital_acquired)
columns_to_remove = [
    "patientunitstayid", "uniquepid", 
    "sepsis_explicit_hospital_acquired", "sepsis_angus_implicit_hospital_acquired", 
    "sepsis_dx_atemporal"
]

# Rimuovi le colonne specificate
df = df.drop(columns=columns_to_remove)
print(f"Colonne dopo la rimozione: {list(df.columns)}")

# Preprocessing: converte valori booleani e categorici
for column in df.columns:
    if df[column].dtype == 'bool':
        df[column] = df[column].astype(int)
    else:
        # Controlla se ci sono valori stringa da convertire
        unique_vals = df[column].dropna().unique()
        if any(isinstance(val, str) for val in unique_vals):
            # Mappa i valori categorici low/medium/high
            if any(val in ['low', 'medium', 'high'] for val in unique_vals if isinstance(val, str)):
                df[column] = df[column].replace({'low': 0, 'medium': 1, 'high': 2})

# Imputazione valori mancanti
imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
df_imputed = pd.DataFrame(imp_mean.fit_transform(df), columns=df.columns)

# Identifica i 5 hospitalid con più dati
top_hospitals = df_imputed['hospitalid'].value_counts().head(5).index.tolist()
print(f"\n5 hospitalid con più dati: {top_hospitals}")

# Per ogni hospitalid, crea train e test set
for hospital_id in top_hospitals:
    print(f"\nProcessing hospitalid: {hospital_id}")
    
    # Filtra i dati per l'hospitalid corrente
    hospital_df = df_imputed[df_imputed['hospitalid'] == hospital_id].copy()
    
    n_rows = len(hospital_df)
    print(f"  Numero di righe per questo ospedale: {n_rows}")
    
    # Se l'ospedale ha più di 3500 righe, riduci a 3500 mantenendo la stratificazione
    if n_rows > 3500:
        # Dividi temporaneamente per stratificazione
        X_temp = hospital_df.drop(columns=["sepsis_hospital_acquired", "hospitalid"])
        y_temp = hospital_df["sepsis_hospital_acquired"]
        
        # Dividi per ottenere 3500 righe in modo stratificato
        X_temp_reduced, _, y_temp_reduced, _ = train_test_split(
            X_temp, y_temp, train_size=3500, stratify=y_temp, random_state=42
        )
        
        # Ricostruisci il dataframe ridotto
        hospital_df_reduced = pd.concat([X_temp_reduced, y_temp_reduced], axis=1)
        hospital_df_reduced["hospitalid"] = hospital_id
        
        print(f"  Ridotto a 3500 righe (rimosse {n_rows - 3500} righe)")
        hospital_df = hospital_df_reduced
    
    # Prepara target e features
    y = hospital_df["sepsis_hospital_acquired"]
    X = hospital_df.drop(columns=["sepsis_hospital_acquired", "hospitalid"])
    
    # Divisione train/test stratificata: train 2500, test 1000
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, train_size=2500, test_size=1000, random_state=42
    )
    
    # Aggiungi la colonna target come ultima colonna
    X_train["sepsis_hospital_acquired"] = y_train
    X_test["sepsis_hospital_acquired"] = y_test
    
    # Salva i file
    train_filename = f"data_fixed_rows/{int(hospital_id)}_train.csv"
    test_filename = f"data_fixed_rows/{int(hospital_id)}_test.csv"
    
    X_train.to_csv(train_filename, index=False)
    X_test.to_csv(test_filename, index=False)
    
    print(f"  File creati: {train_filename} ({len(X_train)} righe)")
    print(f"              {test_filename} ({len(X_test)} righe)")

print(f"\nProcesso completato. File salvati nella cartella 'data_fixed_rows/'")