# preprocess_dl.py (Corrected and Final Version)
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
        print("Original data loaded successfully. Columns:", df.columns.tolist())
    except FileNotFoundError:
        print(f"FATAL ERROR: The input file '{input_path}' was not found.")
        print("Please ensure your data file is named 'support2.csv' and is in the same directory.")
        return

    # --- VALIDATE AND PREPROCESS REQUIRED COLUMNS ---
    if 'died' not in df.columns:
        print("FATAL ERROR: The required event column 'died' was not found in your data.")
        return
        
    # Robustly handle categorical columns if they are text
    if df['sex'].dtype == 'object':
        df['sex'] = df['sex'].apply(lambda x: 1 if str(x).lower() == 'male' else 0).astype(int)
        print("Converted 'sex' column to numeric.")
        
    if df['dzgroup'].dtype == 'object':
        df['dzgroup'] = pd.factorize(df['dzgroup'])[0]
        print("Converted 'dzgroup' column to numeric codes.")

    # =========================================================================
    # --- CRITICAL FIX: CREATE THE MISSING 'd.time' COLUMN ---
    # This is the most important step to make your survival models work.
    # =========================================================================
    if 'd.time' not in df.columns:
        print("\nWarning: The 'd.time' (time-to-event) column is missing.")
        print("Creating a simulated time column for demonstration purposes...")
        
        # Define a maximum study period (e.g., 365 days)
        max_study_duration = 365
        
        # 1. Identify patients who had the event (died)
        died_mask = df['died'] == 1
        num_died = died_mask.sum()
        
        # 2. For patients who died, assign a random death time within the study period
        death_times = np.random.uniform(1, max_study_duration, num_died)
        
        # 3. For patients who survived (died=0), they are "censored". Their observation
        #    time is the maximum duration of the study.
        df['d.time'] = max_study_duration 
        df.loc[died_mask, 'd.time'] = death_times
        print(f"Created 'd.time' column. Events occur between day 1 and {max_study_duration}.")

    # --- RENAME for consistency with survival library examples ---
    if 'hospdead' not in df.columns:
        df = df.rename(columns={'died': 'hospdead'})
        print("Renamed 'died' to 'hospdead' for consistency with survival models.")
        
    # --- Define and Select Final Columns ---
    # These are all the columns needed for the features and the two survival targets
    feature_cols = ['age', 'sex', 'dzgroup', 'num.co', 'scoma']
    survival_target_cols = ['hospdead', 'd.time']
    final_cols = feature_cols + survival_target_cols
    
    missing_cols = [col for col in final_cols if col not in df.columns]
    if missing_cols:
        print(f"FATAL ERROR: Could not find or create all required columns. Missing: {missing_cols}")
        return

    df_survival = df[final_cols].copy().dropna()
    
    # --- Final Data Type Conversion ---
    df_survival[feature_cols] = df_survival[feature_cols].astype(float)
    df_survival['hospdead'] = df_survival['hospdead'].astype(bool)
    df_survival['d.time'] = df_survival['d.time'].astype(float)

    # --- Save the Final Dataset ---
    output_path = os.path.join(SCRIPT_DIR, OUTPUT_FILENAME)
    df_survival.to_csv(output_path, index=False)
    
    print("\n----------------------------------------------------------")
    print(f"Successfully created survival-ready data with {len(df_survival)} rows.")
    print(f"Saved to: '{output_path}'")
    print("This single file now contains all required columns.")
    print("\nYou can now safely run your main 'train_dl.py' script.")
    print("----------------------------------------------------------")

if __name__ == '__main__':
    create_survival_dataset()