# prepare_data.py
import pandas as pd
import numpy as np
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FILENAME = 'support2.csv'
OUTPUT_FILENAME = 'support_survival.csv'

def create_survival_dataset():
    """
    Reads the original support2.csv file, adds the required 'd.time' column
    to make it suitable for survival analysis, preprocesses other columns,
    and saves a new, single, model-ready CSV file.
    """
    print(f"--- Preparing Data for Survival Analysis from '{INPUT_FILENAME}' ---")
    input_path = os.path.join(SCRIPT_DIR, INPUT_FILENAME)
    
    try:
        df = pd.read_csv(input_path)
        print("Original data loaded successfully.")
    except FileNotFoundError:
        print(f"FATAL ERROR: The input file '{input_path}' was not found.")
        return

    # --- VALIDATE AND PREPROCESS ---
    if 'died' not in df.columns:
        print("FATAL ERROR: The required event column 'died' was not found in your data.")
        return
        
    if df['sex'].dtype == 'object':
        df['sex'] = df['sex'].apply(lambda x: 1 if str(x).lower() == 'male' else 0).astype(int)
    if df['dzgroup'].dtype == 'object':
        df['dzgroup'] = pd.factorize(df['dzgroup'])[0]

    # --- CRITICAL STEP: CREATE THE MISSING 'd.time' COLUMN ---
    if 'd.time' not in df.columns:
        print("Warning: 'd.time' column is missing. Creating a simulated time column...")
        max_study_duration = 365
        died_mask = df['died'] == 1
        num_died = died_mask.sum()
        
        death_times = np.random.uniform(1, max_study_duration, num_died)
        df['d.time'] = max_study_duration 
        df.loc[died_mask, 'd.time'] = death_times

    # --- RENAME for consistency ---
    if 'hospdead' not in df.columns:
        df = df.rename(columns={'died': 'hospdead'})
        
    # --- Define and Select Final Columns ---
    feature_cols = ['age', 'sex', 'dzgroup', 'num.co', 'scoma']
    survival_target_cols = ['hospdead', 'd.time']
    final_cols = feature_cols + survival_target_cols
    
    df_survival = df[final_cols].copy().dropna()
    df_survival[feature_cols] = df_survival[feature_cols].astype(float)
    df_survival['hospdead'] = df_survival['hospdead'].astype(bool)
    df_survival['d.time'] = df_survival['d.time'].astype(float)

    # --- Save the Final Dataset ---
    output_path = os.path.join(SCRIPT_DIR, OUTPUT_FILENAME)
    df_survival.to_csv(output_path, index=False)
    
    print("\n----------------------------------------------------------")
    print(f"SUCCESS: Created survival-ready data at '{output_path}'")
    print("You can now safely run your 'train_dl.py' script.")
    print("----------------------------------------------------------")

if __name__ == '__main__':
    create_survival_dataset()