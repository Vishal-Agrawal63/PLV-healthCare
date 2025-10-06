# PATH: /ml/preprocess_dl.py
import pandas as pd
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def preprocess_support_data():
    """
    Reads the local support2.csv file, cleans it using the CORRECT columns, 
    and saves the result.
    """
    print("Loading local SUPPORT dataset...")
    local_file_path = os.path.join(SCRIPT_DIR, 'support2.csv')
    
    try:
        df = pd.read_csv(local_file_path)
        print("Dataset loaded successfully from local file.")
    except FileNotFoundError:
        print(f"FATAL ERROR: 'support2.csv' not found in the '/ml' directory.")
        return

    # --- THIS IS THE CORRECTED LIST OF FEATURES ---
    # These features actually exist in the manually created CSV file.
    features_to_use = [
        'age',
        'sex',
        'dzgroup',
        'num.co',    # Number of comorbidities
        'scoma',     # Glasgow Coma Score
        'died'   # This is our Target variable
    ]

    # Select only the columns that exist, and our target
    df_selected = df[features_to_use].copy()
    
    # Drop rows with any missing values
    df_selected.dropna(inplace=True)
    
    # Convert categorical variables to numeric codes
    if pd.api.types.is_string_dtype(df_selected['sex']):
        df_selected['sex'] = df_selected['sex'].astype('category').cat.codes
    
    if pd.api.types.is_string_dtype(df_selected['dzgroup']):
        df_selected['dzgroup'] = df_selected['dzgroup'].astype('category').cat.codes
    
    # Rename target for clarity
    df_selected.rename(columns={'hospdead': 'died'}, inplace=True)
    
    output_path = os.path.join(SCRIPT_DIR, 'support_cleaned.csv')
    df_selected.to_csv(output_path, index=False)
    
    print(f"Cleaned SUPPORT dataset saved to {output_path}")

if __name__ == '__main__':
    preprocess_support_data()