# train_dl.py (Corrected and Final Version)
import pandas as pd
import os
import joblib

# --- TensorFlow / Keras Imports ---
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# --- Scikit-Survival Imports ---
from sksurv.ensemble import RandomSurvivalForest
from sksurv.util import Surv

# --- PyCox / PyTorch Imports ---
import numpy as np
import torch
from torch import nn
from pycox import models
from pycox.evaluation import EvalSurv 
import torchtuples as tt

# --- Common Imports ---
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE_NAME = 'support_survival.csv'

# ===================================================================
# --- FIX: HELPER CLASS FOR COX-TIME MODEL ---
# This special network handles the two inputs (features + time)
# required by the CoxTime model.
# ===================================================================
class CoxTimeMLP(nn.Module):
    """A simple MLP that accepts features and time, then concatenates them."""
    def __init__(self, in_features, hidden_dims=[64, 32]):
        super().__init__()
        # We add 1 to in_features for the concatenated time input
        all_dims = [in_features + 1] + hidden_dims + [1]
        layers = []
        for i in range(len(all_dims) - 2):
            layers.extend([
                nn.Linear(all_dims[i], all_dims[i+1]),
                nn.ReLU(),
                nn.Dropout(0.3)
            ])
        layers.append(nn.Linear(all_dims[-2], all_dims[-1]))
        self.network = nn.Sequential(*layers)
        
    def forward(self, x_features, x_time):
        # This is the important part: it accepts two arguments
        x_combined = torch.cat([x_features, x_time], dim=1)
        return self.network(x_combined)
# ===================================================================

def get_pca_data(save_preprocessors=False):
    """
    Loads the survival data, applies scaling and PCA, and returns split datasets.
    """
    print("--- Loading data and preparing PCA components ---")
    input_path = os.path.join(SCRIPT_DIR, DATA_FILE_NAME)
    try:
        df = pd.read_csv(input_path)
    except FileNotFoundError:
        print(f"FATAL ERROR: '{input_path}' not found. Please run 'prepare_data.py' first!")
        exit()

    features = [col for col in df.columns if col not in ['hospdead', 'd.time']]
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    pca = PCA(n_components=0.95)
    
    x_train_scaled = scaler.fit_transform(df_train[features])
    x_train_pca = pca.fit_transform(x_train_scaled).astype('float32')
    x_test_pca = pca.transform(df_test[features]).astype('float32')

    print(f"PCA reduced {len(features)} features to {pca.n_components_} components.")

    if save_preprocessors:
        joblib.dump(scaler, os.path.join(SCRIPT_DIR, 'universal_pca_scaler.pkl'))
        joblib.dump(pca, os.path.join(SCRIPT_DIR, 'universal_pca_transformer.pkl'))
        print("Universal scaler and PCA transformer saved.")
        
    return df_train, df_test, x_train_pca, x_test_pca

# ===================================================================
# MODEL TRAINING FUNCTIONS (Final Versions)
# ===================================================================
def train_keras_pca_model(df_train, df_test, x_train_pca, x_test_pca):
    print("\n--- Training Keras + PCA Model ---")
    y_train = df_train['hospdead']
    y_test = df_test['hospdead']
    
    model = Sequential([
        Dense(64, activation='relu', input_shape=(x_train_pca.shape[1],)), Dropout(0.3),
        Dense(32, activation='relu'), Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(x_train_pca, y_train, epochs=50, batch_size=32, verbose=0) 
    loss, acc = model.evaluate(x_test_pca, y_test, verbose=0)
    print(f"Evaluation: Accuracy={acc:.4f}")
    model.save(os.path.join(SCRIPT_DIR, 'keras_pca_model.h5'))
    print("Keras+PCA model saved.")

def train_rsf_pca_model(df_train, df_test, x_train_pca, x_test_pca):
    print("\n--- Training Random Survival Forest + PCA Model ---")
    y_train = Surv.from_dataframe('hospdead', 'd.time', df_train)
    y_test = Surv.from_dataframe('hospdead', 'd.time', df_test)
    
    rsf = RandomSurvivalForest(n_estimators=100, min_samples_leaf=15, n_jobs=-1, random_state=42)
    rsf.fit(x_train_pca, y_train)
    score = rsf.score(x_test_pca, y_test)
    print(f"Evaluation (Concordance Index): {score:.4f}")
    joblib.dump(rsf, os.path.join(SCRIPT_DIR, 'rsf_pca_model.pkl'))
    print("RSF+PCA model saved.")

def train_deephit_pca_model(df_train, df_test, x_train_pca, x_test_pca):
    print("\n--- Training DeepHit + PCA Model ---")
    get_target = lambda df: (df['d.time'].values, df['hospdead'].values)
    labtrans = models.DeepHitSingle.label_transform(10)
    y_train = labtrans.fit_transform(*get_target(df_train))

    net = nn.Sequential(
        nn.Linear(x_train_pca.shape[1], 64), nn.ReLU(), nn.Dropout(0.3),
        nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.3),
        nn.Linear(32, labtrans.out_features)
    )
    model = models.DeepHitSingle(net, tt.optim.Adam, alpha=0.2, sigma=0.1, duration_index=labtrans.cuts)
    model.fit(x_train_pca, y_train, batch_size=128, epochs=100, verbose=False, callbacks=[tt.callbacks.EarlyStopping()])
    print("Model fitting complete.")
    model.save_model_weights(os.path.join(SCRIPT_DIR, 'deephit_pca_model.pt'))
    joblib.dump(labtrans, os.path.join(SCRIPT_DIR, 'deephit_pca_labtrans.pkl'))
    print("DeepHit+PCA model and labtrans saved.")
    
def train_deepsurv_pca_model(df_train, df_test, x_train_pca, x_test_pca):
    print("\n--- Training DeepSurv + PCA Model ---")
    get_target = lambda df: (df['d.time'].values, df['hospdead'].values)
    y_train = get_target(df_train)
    y_test = get_target(df_test)
    
    net = nn.Sequential(
        nn.Linear(x_train_pca.shape[1], 64), nn.ReLU(), nn.Dropout(0.3),
        nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.3),
        nn.Linear(32, 1)
    )
    model = models.CoxPH(net, tt.optim.Adam)
    model.fit(x_train_pca, y_train, batch_size=128, epochs=100, verbose=False, callbacks=[tt.callbacks.EarlyStopping()])
    _ = model.compute_baseline_hazards()
    surv_df = model.predict_surv_df(x_test_pca)
    
    y_test_durations, y_test_events = y_test
    ev = EvalSurv(surv_df, y_test_durations, y_test_events, censor_surv='km')
    c_index = ev.concordance_td()
    print(f"Evaluation (Concordance Index): {c_index:.4f}")
    model.save_model_weights(os.path.join(SCRIPT_DIR, 'deepsurv_pca_model.pt'))
    print("DeepSurv+PCA model saved.")

def train_coxtime_pca_model(df_train, df_test, x_train_pca, x_test_pca):
    print("\n--- Training Cox-Time + PCA Model ---")
    get_target = lambda df: (df['d.time'].values, df['hospdead'].values)
    labtrans = models.CoxTime.label_transform()
    y_train = labtrans.fit_transform(*get_target(df_train))
    y_test = get_target(df_test)

    # --- FIX: Use the special CoxTimeMLP network ---
    net = CoxTimeMLP(in_features=x_train_pca.shape[1])
    # ---
    
    model = models.CoxTime(net, tt.optim.Adam, labtrans=labtrans)
    model.fit(x_train_pca, y_train, batch_size=128, epochs=100, verbose=False, callbacks=[tt.callbacks.EarlyStopping()])
    
    _ = model.compute_baseline_hazards()
    surv_df = model.predict_surv_df(x_test_pca)
    y_test_durations, y_test_events = y_test
    ev = EvalSurv(surv_df, y_test_durations, y_test_events, censor_surv='km')
    c_index = ev.concordance_td()
    
    print(f"Evaluation (Concordance Index): {c_index:.4f}")
    model.save_model_weights(os.path.join(SCRIPT_DIR, 'coxtime_pca_model.pt'))
    joblib.dump(labtrans, os.path.join(SCRIPT_DIR, 'coxtime_pca_labtrans.pkl'))
    print("Cox-Time+PCA model and labtrans saved.")
    
# ===================================================================
# MAIN EXECUTION BLOCK
# ===================================================================
if __name__ == '__main__':
    print("Starting unified PCA-based training process...")
    
    df_train, df_test, x_train_pca, x_test_pca = get_pca_data(save_preprocessors=True)
    
    train_keras_pca_model(df_train, df_test, x_train_pca, x_test_pca)
    train_rsf_pca_model(df_train, df_test, x_train_pca, x_test_pca)
    train_deephit_pca_model(df_train, df_test, x_train_pca, x_test_pca)
    train_deepsurv_pca_model(df_train, df_test, x_train_pca, x_test_pca)
    train_coxtime_pca_model(df_train, df_test, x_train_pca, x_test_pca)
    
    print("\n\nAll PCA-based models have been trained successfully!")