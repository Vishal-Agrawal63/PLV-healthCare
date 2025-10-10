import pandas as pd
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def create_survival_dataset():
    """
    Reads the original support2.csv file, selects the necessary features
    for survival analysis, handles categorical data and missing values,
    and saves a new, model-ready CSV.
    """
    try:
        input_path = os.path.join(SCRIPT_DIR, 'support2.csv')
        df = pd.read_csv(input_path)
    except FileNotFoundError:
        print(f"Error: '{input_path}' not found. Please ensure it is in the /ml directory.")
        return

    # --- Preprocessing Categorical Features ---
    
    # 1. Convert 'sex' column from text ('male'/'female') to numbers (1/0)
    df['sex'] = df['sex'].apply(lambda x: 1 if x == 'male' else 0)
    
    # 2. --- FIX: Convert categorical 'dzgroup' to numerical codes ---
    # This is the solution to the ValueError.
    # pd.factorize() automatically assigns a unique integer (0, 1, 2, ...) to each unique text category.
    df['dzgroup'] = pd.factorize(df['dzgroup'])[0]

    # --- Define Columns for the Survival Dataset ---
    
    # Select the features that the models will use for prediction
    feature_cols = ['age', 'sex', 'dzgroup', 'num.co', 'scoma']
    
    # 'hospdead' is the event indicator (1 if died in hospital, 0 if survived/censored)
    event_col = 'hospdead'
    
    # 'd.time' is the time to event (death) or censoring
    time_col = 'd.time'
    
    # Create a list of all columns we need to keep
    required_cols = feature_cols + [event_col, time_col]
    
    # Create the new dataframe, keeping only the required columns and dropping any rows
    # that have missing values in these specific columns.
    df_survival = df[required_cols].dropna().copy()
    
    # --- Final Data Type Conversion ---
    
    # Ensure all feature columns are numeric (float)
    df_survival[feature_cols] = df_survival[feature_cols].astype(float)
    
    # Ensure the event indicator is a boolean (True/False)
    df_survival[event_col] = df_survival[event_col].astype(bool)
    
    # Ensure the time column is numeric (float)
    df_survival[time_col] = df_survival[time_col].astype(float)

    # --- Save the Final Dataset ---
    output_path = os.path.join(SCRIPT_DIR, 'support_survival.csv')
    df_survival.to_csv(output_path, index=False)
    
    print(f"Successfully created survival dataset with {len(df_survival)} rows.")
    print(f"Saved to '{output_path}'")

if __name__ == '__main__':
    create_survival_dataset()