# Unified Prediction API
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
import os
import torch
from torch import nn

# --- Import ML Libraries ---
from tensorflow.keras.models import load_model as load_keras_model
from pycox import models as pycox_models
from sksurv.ensemble import RandomSurvivalForest 
from sklearn.compose import ColumnTransformer 

app = Flask(__name__)
CORS(app)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# --- Global dictionary to hold all loaded models and preprocessors ---
loaded_artifacts = {}

def load_all_models():
    """Load all available models and their preprocessors into memory on startup."""
    
    # --- 1. Load Keras+PCA Model ---
    try:
        model = load_keras_model(os.path.join(SCRIPT_DIR, 'keras_pca_model.h5'))
        scaler = joblib.load(os.path.join(SCRIPT_DIR, 'keras_pca_scaler.pkl'))
        pca = joblib.load(os.path.join(SCRIPT_DIR, 'keras_pca_pca.pkl'))
        loaded_artifacts['keras_pca'] = {'model': model, 'scaler': scaler, 'pca': pca}
        print("Keras+PCA model loaded successfully.")
    except Exception as e:
        print(f"Warning: Could not load Keras+PCA model. {e}")

    # --- 2. Load RSF Model ---
    try:
        model = joblib.load(os.path.join(SCRIPT_DIR, 'rsf_model.pkl'))
        loaded_artifacts['rsf'] = {'model': model}
        print("RSF model loaded successfully.")
    except Exception as e:
        print(f"Warning: Could not load RSF model. {e}")
        
    # --- 3. Load DeepHit Model (CORRECTED) ---
    try:
        # --- FIX 1: Load the correct preprocessor file ---
        preprocessor = joblib.load(os.path.join(SCRIPT_DIR, 'deephit_preprocessor.pkl'))
        labtrans = joblib.load(os.path.join(SCRIPT_DIR, 'deephit_labtrans.pkl'))
        
        # A robust way to determine the number of input features after transformation
        in_features = sum(len(cols) for name, trans, cols in preprocessor.transformers_ if trans != 'drop')

        out_features = labtrans.out_features
        net = nn.Sequential(
            nn.Linear(in_features, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(32, out_features)
        )
        model = pycox_models.DeepHitSingle(net, duration_index=labtrans.cuts)
        
        # --- FIX 2: Use the correct method to load weights ---
        model.load_model_weights(os.path.join(SCRIPT_DIR, 'deephit_model.pt'))
        
        # --- FIX 3: Store the preprocessor with the correct key ---
        loaded_artifacts['deephit'] = {'model': model, 'preprocessor': preprocessor, 'labtrans': labtrans}
        print("DeepHit model loaded successfully.")
    except Exception as e:
        print(f"Warning: Could not load DeepHit model. {e}")

@app.route('/predict_survival', methods=['POST'])
def predict():
    data = request.get_json()
    if not data or 'model_name' not in data or 'patient_data' not in data:
        return jsonify({'error': 'Invalid payload. "model_name" and "patient_data" are required.'}), 400
    
    model_name = data['model_name']
    patient_data = data['patient_data']
    
    if model_name not in loaded_artifacts:
        return jsonify({'error': f"Model '{model_name}' not found or failed to load."}), 404

    try:
        input_df = pd.DataFrame([patient_data])
        
        # --- Route to the correct prediction logic based on model_name ---
        if model_name == 'keras_pca':
            artifacts = loaded_artifacts['keras_pca']
            # The keras model expects specific column names from its training data,
            # which might differ from the survival data.
            keras_cols = artifacts['scaler'].feature_names_in_
            input_df_keras = input_df[keras_cols]
            
            scaled_features = artifacts['scaler'].transform(input_df_keras)
            pca_features = artifacts['pca'].transform(scaled_features)
            prediction_prob = artifacts['model'].predict(pca_features)[0][0]
            survival_probability = 1 - prediction_prob

        elif model_name == 'rsf':
            artifacts = loaded_artifacts['rsf']
            survival_funcs = artifacts['model'].predict_survival_function(input_df)
            survival_probability = survival_funcs[0](30) # Get probability at 30 days

        elif model_name == 'deephit':
            # --- FIX 4: Use the correct preprocessor key ---
            artifacts = loaded_artifacts['deephit']
            processed_input = artifacts['preprocessor'].transform(input_df).astype('float32')
            surv_df = artifacts['model'].predict_surv_df(processed_input)
            
            prediction_time = 30
            idx = np.searchsorted(surv_df.columns, prediction_time, side='right') - 1
            survival_probability = surv_df.iloc[0, max(0, idx)]
            
        else:
            return jsonify({'error': 'Invalid model name specified.'}), 400

        return jsonify({'survival_probability': round(float(survival_probability) * 100, 2)})
        
    except Exception as e:
        return jsonify({'error': f'An error occurred during prediction: {str(e)}'}), 500

# --- Load models on startup ---
load_all_models()

if __name__ == '__main__':
    app.run(port=5001, debug=True)