# PATH: /ml/train_dl.py
import pandas as pd
import os
import joblib

# --- Keras / TensorFlow Imports ---
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# --- Scikit-Survival (RSF) Imports ---
from sksurv.ensemble import RandomSurvivalForest
from sksurv.util import Surv

# --- PyCox Imports ---
import numpy as np
import torch
from torch import nn
from pycox import models
import torchtuples as tt

# --- Common Imports ---
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.compose import ColumnTransformer

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ===================================================================
# Custom Network for CoxTime to handle multiple inputs
# ===================================================================
class MLP_CoxTime(nn.Module):
    """
    A custom MLP class is needed for CoxTime because its `fit` method
    passes both patient features (x) and time information (t) to the
    network's forward pass. A standard nn.Sequential only accepts one input.

    This class defines a `forward` method that accepts both `x` and `t`,
    but only passes `x` to the sequential layers, which is the expected
    behavior for this model architecture.
    """
    def __init__(self, in_features, num_nodes, out_features, batch_norm=False, dropout=0.1):
        super().__init__()
        # Simplified layer creation
        layers = []
        for n_in, n_out in zip([in_features] + num_nodes, num_nodes + [out_features]):
            layers.append(nn.Linear(n_in, n_out))
            layers.append(nn.ReLU())
            if batch_norm:
                layers.append(nn.BatchNorm1d(n_out))
            if dropout:
                layers.append(nn.Dropout(dropout))
        # Remove the last activation, as is typical for regression outputs
        layers = layers[:-1]
        self.net = nn.Sequential(*layers)

    def forward(self, x, t):
        # The key fix: This forward method accepts both x and t,
        # but the network only processes x.
        return self.net(x)


# ===================================================================
# MODEL 1: KERAS + PCA (Your Original Model)
# ===================================================================
def train_keras_pca_model():
    """
    Trains a Keras binary classifier on the 'died' column, using PCA.
    Saves the model, scaler, and PCA transformer.
    """
    print("==============================================")
    print("--- Training Keras + PCA Model ---")
    print("==============================================")
    print("Loading original cleaned SUPPORT data...")
    input_path = os.path.join(SCRIPT_DIR, 'support_cleaned.csv')
    try:
        df = pd.read_csv(input_path)
    except FileNotFoundError:
        print(f"Error: '{input_path}' not found.")
        return

    X = df.drop('died', axis=1)
    y = df['died']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"\nApplying PCA to the {X.shape[1]} features...")
    pca = PCA(n_components=0.95)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    
    print(f"PCA reduced features from {X.shape[1]} to {pca.n_components_}")
    
    model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train_pca.shape[1],)), Dropout(0.3),
        Dense(32, activation='relu'), Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    print("\nTraining model on PCA components...")
    model.fit(X_train_pca, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=0) 
    
    loss, accuracy = model.evaluate(X_test_pca, y_test, verbose=0)
    print(f"\nModel Evaluation: Loss={loss:.4f}, Accuracy={accuracy:.4f}")
    
    model.save(os.path.join(SCRIPT_DIR, 'keras_pca_model.h5'))
    joblib.dump(scaler, os.path.join(SCRIPT_DIR, 'keras_pca_scaler.pkl'))
    joblib.dump(pca, os.path.join(SCRIPT_DIR, 'keras_pca_pca.pkl'))
    print("\nSuccessfully saved Keras model, scaler, and PCA transformer.")

# ===================================================================
# MODEL 2: RANDOM SURVIVAL FOREST (RSF)
# ===================================================================
def train_rsf_model():
    """
    Trains a Random Survival Forest model on the time-to-event data.
    Saves the trained model.
    """
    print("\n\n==============================================")
    print("--- Training Random Survival Forest Model ---")
    print("==============================================")
    print("Loading survival data...")
    input_path = os.path.join(SCRIPT_DIR, 'support_survival.csv')
    try:
        df = pd.read_csv(input_path)
    except FileNotFoundError:
        print(f"Error: '{input_path}' not found. Run prepare_survival_data.py first.")
        return

    X = df.drop(['hospdead', 'd.time'], axis=1)
    y = Surv.from_dataframe('hospdead', 'd.time', df)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    rsf = RandomSurvivalForest(n_estimators=100, min_samples_split=10, min_samples_leaf=15, n_jobs=-1, random_state=42)
    
    print("Fitting RSF model...")
    rsf.fit(X_train, y_train)
    
    score = rsf.score(X_test, y_test)
    print(f"\nModel Evaluation (Concordance Index): {score:.4f}")
    
    joblib.dump(rsf, os.path.join(SCRIPT_DIR, 'rsf_model.pkl'))
    print(f"RSF model saved to rsf_model.pkl")

# ===================================================================
# MODEL 3: DEEPHIT
# ===================================================================
def train_deephit_model():
    """
    Trains a DeepHit survival model on the time-to-event data.
    Saves the model weights and necessary preprocessors.
    """
    print("\n\n==============================================")
    print("--- Training DeepHit Model ---")
    print("==============================================")
    print("Loading survival data...")
    input_path = os.path.join(SCRIPT_DIR, 'support_survival.csv')
    try:
        df = pd.read_csv(input_path)
    except FileNotFoundError:
        print(f"Error: '{input_path}' not found. Run prepare_survival_data.py first.")
        return
    
    df_train = df.sample(frac=0.8, random_state=42)
    df_val = df_train.sample(frac=0.2, random_state=123)
    
    cols_standardize = ['age', 'num.co', 'scoma']
    cols_leave = ['sex', 'dzgroup']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), cols_standardize),
            ('cat', 'passthrough', cols_leave)
        ])
    
    x_train = preprocessor.fit_transform(df_train).astype('float32')
    x_val = preprocessor.transform(df_val).astype('float32')
    
    num_durations = 10
    labtrans = models.DeepHitSingle.label_transform(num_durations)
    
    get_target = lambda df: (df['d.time'].values, df['hospdead'].values)
    y_train = labtrans.fit_transform(*get_target(df_train))
    y_val = labtrans.transform(*get_target(df_val))

    in_features = x_train.shape[1]
    out_features = labtrans.out_features
    
    net = nn.Sequential(
        nn.Linear(in_features, 64), nn.ReLU(), nn.Dropout(0.3),
        nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.3),
        nn.Linear(32, out_features)
    )
    
    model = models.DeepHitSingle(net, tt.optim.Adam, alpha=0.2, sigma=0.1, duration_index=labtrans.cuts)
    
    batch_size, epochs = 64, 100
    callbacks = [tt.callbacks.EarlyStopping()]
    
    print("\nFitting DeepHit model...")
    model.fit(x_train, y_train, batch_size, epochs, callbacks, val_data=(x_val, y_val), verbose=False)
    model.save_model_weights(os.path.join(SCRIPT_DIR, 'deephit_model.pt'))
    
    joblib.dump(preprocessor, os.path.join(SCRIPT_DIR, 'deephit_preprocessor.pkl'))
    joblib.dump(labtrans, os.path.join(SCRIPT_DIR, 'deephit_labtrans.pkl'))
    print("\nSuccessfully saved DeepHit model and preprocessors.")

# ===================================================================
# MODEL 4: COX-TIME
# ===================================================================
def train_coxtime_model():
    """
    Trains a Cox-Time survival model, which can model time-varying effects.
    Saves the model weights and necessary preprocessors.
    """
    print("\n\n==============================================")
    print("--- Training Cox-Time Model ---")
    print("==============================================")
    print("Loading survival data...")
    input_path = os.path.join(SCRIPT_DIR, 'support_survival.csv')
    try:
        df = pd.read_csv(input_path)
    except FileNotFoundError:
        print(f"Error: '{input_path}' not found. Run prepare_survival_data.py first.")
        return

    df_train = df.sample(frac=0.8, random_state=42)
    # Note: df_val is created from df_train, not the original df. This is intentional.
    df_val = df_train.sample(frac=0.2, random_state=123)
    df_train = df_train.drop(df_val.index) # Ensure validation set is not in training set

    cols_standardize = ['age', 'num.co', 'scoma']
    cols_leave = ['sex', 'dzgroup']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), cols_standardize),
            ('cat', 'passthrough', cols_leave)
        ])

    x_train = preprocessor.fit_transform(df_train).astype('float32')
    
    labtrans = models.CoxTime.label_transform()
    
    get_target = lambda df: (df['d.time'].values, df['hospdead'].values)
    y_train = labtrans.fit_transform(*get_target(df_train))

    in_features = x_train.shape[1]
    
    net = MLP_CoxTime(
        in_features=in_features,
        num_nodes=[64, 32],
        out_features=1,
        batch_norm=False,
        dropout=0.3
    )
    
    model = models.CoxTime(net, tt.optim.Adam)
    
    batch_size, epochs = 64, 100
    callbacks = [tt.callbacks.EarlyStopping()]
    
    # We still need the original (unsorted) x_val and y_val for the fitting process
    x_val = preprocessor.transform(df_val).astype('float32')
    y_val = labtrans.transform(*get_target(df_val))
    
    print("\nFitting Cox-Time model...")
    log = model.fit(x_train, y_train, batch_size, epochs, callbacks, val_data=(x_val, y_val), verbose=False)

    print("Computing baseline hazards for Cox-Time model...")
    
    # --- THIS IS THE FIX ---
    # 1. Sort the validation dataframe by the duration column ('d.time').
    df_val_sorted = df_val.sort_values(by='d.time')
    
    # 2. Re-transform the features and target based on this sorted dataframe.
    x_val_sorted = preprocessor.transform(df_val_sorted).astype('float32')
    y_val_sorted = labtrans.transform(*get_target(df_val_sorted))

    # 3. Pass the new, sorted data to compute_baseline_hazards.
    baseline_hazards = model.compute_baseline_hazards(x_val_sorted, y_val_sorted)
    # -----------------------

    model.save_model_weights(os.path.join(SCRIPT_DIR, 'coxtime_model.pt'))
    joblib.dump(baseline_hazards, os.path.join(SCRIPT_DIR, 'coxtime_baseline_hazards.pkl'))
    joblib.dump(preprocessor, os.path.join(SCRIPT_DIR, 'coxtime_preprocessor.pkl'))
    joblib.dump(labtrans, os.path.join(SCRIPT_DIR, 'coxtime_labtrans.pkl'))
    print("\nSuccessfully saved Cox-Time model, preprocessors, and baseline hazards.")

# Place this function inside train_dl.py

# ===================================================================
# MODEL 5: DEEPSURV (CoxPH) - THIS IS THE FULLY CORRECTED VERSION
# ===================================================================
def train_deepsurv_model():
    """
    Trains a DeepSurv (CoxPH) model and saves the full preprocessor object.
    """
    print("\n\n==============================================")
    print("--- Training DeepSurv (CoxPH) Model ---")
    print("==============================================")
    print("Loading survival data...")
    input_path = os.path.join(SCRIPT_DIR, 'support_survival.csv')
    try:
        df = pd.read_csv(input_path)
    except FileNotFoundError:
        print(f"Error: '{input_path}' not found. Run prepare_survival_data.py first.")
        return

    df_train = df.sample(frac=0.8, random_state=42)
    df_val = df_train.sample(frac=0.2, random_state=123)
    df_train = df_train.drop(df_val.index)

    # --- THIS IS THE CORRECT PREPROCESSING LOGIC ---
    # Define which columns get which treatment
    cols_standardize = ['age', 'num.co', 'scoma']
    categorical_cols = ['sex', 'dzgroup']

    # Create the preprocessor object
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), cols_standardize),
            ('cat', OneHotEncoder(handle_unknown='ignore', drop='first'), categorical_cols)
        ]
    )

    # Separate features (X) from the target (y)
    X_train_df = df_train.drop(['d.time', 'hospdead'], axis=1)
    x_train = preprocessor.fit_transform(X_train_df).astype('float32')
    X_val_df = df_val.drop(['d.time', 'hospdead'], axis=1)
    x_val = preprocessor.transform(X_val_df).astype('float32')
    # ----------------------------------------------

    labtrans = models.CoxTime.label_transform()
    get_target = lambda df: (df['d.time'].values, df['hospdead'].values)
    y_train = labtrans.fit_transform(*get_target(df_train))
    y_val = labtrans.transform(*get_target(df_val))

    in_features = x_train.shape[1]

    net = nn.Sequential(
        nn.Linear(in_features, 64), nn.ReLU(), nn.Dropout(0.3),
        nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.3),
        nn.Linear(32, 1)
    )

    model = models.CoxPH(net, tt.optim.Adam)
    batch_size, epochs = 64, 100
    callbacks = [tt.callbacks.EarlyStopping()]

    print("\nFitting DeepSurv model...")
    model.fit(x_train, y_train, batch_size, epochs, callbacks, val_data=(x_val, y_val), verbose=False)

    print("Computing baseline hazards for DeepSurv model...")
    df_val_sorted = df_val.sort_values(by='d.time')
    X_val_sorted_df = df_val_sorted.drop(['d.time', 'hospdead'], axis=1)
    x_val_sorted = preprocessor.transform(X_val_sorted_df).astype('float32')
    y_val_sorted = labtrans.transform(*get_target(df_val_sorted))
    baseline_hazards = model.compute_baseline_hazards(x_val_sorted, y_val_sorted)

    # --- SAVE THE CORRECT ARTIFACTS ---
    model.save_model_weights(os.path.join(SCRIPT_DIR, 'deepsurv_model.pt'))
    # This line now correctly saves the preprocessor object
    joblib.dump(preprocessor, os.path.join(SCRIPT_DIR, 'deepsurv_preprocessor.pkl'))
    joblib.dump(baseline_hazards, os.path.join(SCRIPT_DIR, 'deepsurv_baseline_hazards.pkl'))
    joblib.dump(labtrans, os.path.join(SCRIPT_DIR, 'deepsurv_labtrans.pkl'))
    joblib.dump(df_val, os.path.join(SCRIPT_DIR, 'deepsurv_df_val.pkl')) 
    print("\nSuccessfully saved DeepSurv model, PREPROCESSOR, and baseline hazards.")

# ===================================================================
# MAIN EXECUTION BLOCK
# ===================================================================
if __name__ == '__main__':
    print("Starting unified training process for all models...")
    
    train_keras_pca_model()
    train_rsf_model()
    train_deephit_model()
    train_coxtime_model()
    train_deepsurv_model()

    print("\n\nAll models have been trained successfully!")