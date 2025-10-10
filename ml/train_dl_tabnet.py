# PATH: /ml/train_dl_tabnet.py (NEW PCA VERSION)

import pandas as pd
import numpy as np
import os
import joblib # <-- Add Joblib for saving the pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA # <-- Import PCA
from sklearn.pipeline import Pipeline # <-- Import Pipeline
from pytorch_tabnet.tab_model import TabNetClassifier
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def train_tabnet_with_pca():
    print("Loading cleaned SUPPORT data for PyTorch TabNet with PCA...")
    input_path = os.path.join(SCRIPT_DIR, 'support_cleaned.csv')
    
    try:
        df = pd.read_csv(input_path)
    except FileNotFoundError:
        print(f"Error: '{input_path}' not found. Please run preprocess_dl.py first.")
        return

    X = df.drop('died', axis=1)
    y = df['died']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # --- Step 1: Create a PCA Pipeline ---
    # This pipeline will first scale the data, then apply PCA to reduce it to 3 dimensions.
    print("Creating and fitting the PCA pipeline...")
    pca_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=3, random_state=RANDOM_STATE)) 
    ])
    
    # Fit the pipeline on the training data and transform it.
    X_train_transformed = pca_pipeline.fit_transform(X_train)
    X_test_transformed = pca_pipeline.transform(X_test)
    
    print(f"Original feature shape: {X_train.shape}")
    print(f"Shape after PCA transformation: {X_train_transformed.shape}")

    # Save the fitted pipeline so the API can use it later.
    pipeline_path = os.path.join(SCRIPT_DIR, 'pca_pipeline.pkl')
    joblib.dump(pca_pipeline, pipeline_path)
    print(f"PCA pipeline saved to {pipeline_path}")

    # --- Step 2: Train TabNet on the Transformed Data ---
    y_train_np = y_train.to_numpy()
    y_test_np = y_test.to_numpy()

    model = TabNetClassifier(
        optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=2e-2),
        device_name='cuda' if torch.cuda.is_available() else 'cpu' 
    )
    
    print("\n--- Training TabNet model on PCA-transformed data ---")
    model.fit(
        X_train=X_train_transformed, y_train=y_train_np,
        eval_set=[(X_test_transformed, y_test_np)],
        patience=5,
        max_epochs=50,
        eval_metric=['accuracy']
    )
    
    model_path = os.path.join(SCRIPT_DIR, 'survival_model_tabnet_pca')
    saved_path = model.save_model(model_path)
    print(f"\nTabNet+PCA model saved to {saved_path}")

if __name__ == '__main__':
    train_tabnet_with_pca()