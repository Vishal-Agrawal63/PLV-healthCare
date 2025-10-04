# PATH: /ml/preprocess.py
import pandas as pd
import numpy as np
import os 

try:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    SCRIPT_DIR = os.getcwd()

def preprocess_data(file_path=os.path.join(SCRIPT_DIR, 'h216.csv')):
    """
    Cleans and preprocesses the MEPS HC-216 dataset.
    """
    print(f"Attempting to load data from: {file_path}") # Added a debug print
    
    # Check if the file actually exists before trying to read it
    if not os.path.exists(file_path):
        print(f"Error: File not found at the specified path: {file_path}")
        print("Please ensure 'h216.csv' is in the same directory as this script.")
        return # Exit the function early

    df = pd.read_csv(file_path)
    print("Dataset loaded successfully.")

    # --- (Rest of the function is the same) ---
    feature_map = {
        'DUPERSID': 'PersonID', 'AGE19X': 'Age', 'SEX': 'Sex',
        'RACETHX': 'Race', 'POVCAT19': 'PovertyCategory', 'INSCOV19': 'InsuranceCoverage',
        'RTHLTH53': 'HealthStatus', 'OBTOTV19': 'OfficeVisits', 'OPTOTV19': 'OutpatientVisits',
        'ERTOT19': 'ERVisits', 'IPDIS19': 'HospitalDischarges', 'TOTEXP19': 'TotalExpenditure'
    }
    
    features = list(feature_map.keys())
    df_selected = df[features].copy()
    df_selected.rename(columns=feature_map, inplace=True)

    print("Handling missing and invalid values...")
    for col in df_selected.columns:
        if df_selected[col].dtype in ['int64', 'float64']:
            df_selected[col] = df_selected[col].apply(lambda x: np.nan if x < 0 else x)

    for col in ['Age', 'OfficeVisits', 'OutpatientVisits', 'ERVisits', 'HospitalDischarges']:
        median_val = df_selected[col].median()
        df_selected[col].fillna(median_val, inplace=True)

    for col in ['Sex', 'Race', 'PovertyCategory', 'InsuranceCoverage', 'HealthStatus']:
         mode_val = df_selected[col].mode()[0]
         df_selected[col].fillna(mode_val, inplace=True)

    df_selected.dropna(subset=['TotalExpenditure'], inplace=True)
    print("Encoding categorical variables...")
    categorical_cols = ['Sex', 'Race', 'PovertyCategory', 'InsuranceCoverage', 'HealthStatus']
    df_processed = pd.get_dummies(df_selected, columns=categorical_cols, drop_first=True)

    print("Saving cleaned dataset...")
    output_path = os.path.join(SCRIPT_DIR, 'patients_cleaned.csv')
    df_processed.to_csv(output_path, index=False)
    
    print(f"Preprocessing complete. Cleaned data saved to '{output_path}'.")

if __name__ == '__main__':
    preprocess_data()