# PATH: /ml/api_dl.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
import os
import torch
from torch import nn
import torchtuples as tt

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

# ===================================================================
# Custom Network for CoxTime to handle multiple inputs
# THIS MUST BE IDENTICAL TO THE ONE IN THE TRAINING SCRIPT
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
        layers = []
        for n_in, n_out in zip([in_features] + num_nodes, num_nodes + [out_features]):
            layers.append(nn.Linear(n_in, n_out))
            layers.append(nn.ReLU())
            if batch_norm:
                layers.append(nn.BatchNorm1d(n_out))
            if dropout:
                layers.append(nn.Dropout(dropout))
        layers = layers[:-1]
        self.net = nn.Sequential(*layers)

    def forward(self, x, t):
        return self.net(x)

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
        
    # --- 3. Load DeepHit Model ---
    try:
        preprocessor = joblib.load(os.path.join(SCRIPT_DIR, 'deephit_preprocessor.pkl'))
        labtrans = joblib.load(os.path.join(SCRIPT_DIR, 'deephit_labtrans.pkl'))
        
        in_features = sum(len(cols) for name, trans, cols in preprocessor.transformers_ if trans != 'drop')
        out_features = labtrans.out_features
        net = nn.Sequential(
            nn.Linear(in_features, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(32, out_features)
        )
        model = pycox_models.DeepHitSingle(net, duration_index=labtrans.cuts)
        
        model.load_model_weights(os.path.join(SCRIPT_DIR, 'deephit_model.pt'))
        
        loaded_artifacts['deephit'] = {'model': model, 'preprocessor': preprocessor, 'labtrans': labtrans}
        print("DeepHit model loaded successfully.")
    except Exception as e:
        print(f"Warning: Could not load DeepHit model. {e}")
    
    # --- 4. Load Cox-Time Model ---
    try:
        preprocessor = joblib.load(os.path.join(SCRIPT_DIR, 'coxtime_preprocessor.pkl'))
        labtrans = joblib.load(os.path.join(SCRIPT_DIR, 'coxtime_labtrans.pkl'))
        baseline_hazards = joblib.load(os.path.join(SCRIPT_DIR, 'coxtime_baseline_hazards.pkl'))

        in_features = sum(len(cols) for name, trans, cols in preprocessor.transformers_ if trans != 'drop')
        
        net = MLP_CoxTime(
            in_features=in_features,
            num_nodes=[64, 32],
            out_features=1,
            batch_norm=False,
            dropout=0.3
        )
        model = pycox_models.CoxTime(net)

        model.load_model_weights(os.path.join(SCRIPT_DIR, 'coxtime_model.pt'))
        
        loaded_artifacts['coxtime'] = {
            'model': model, 
            'preprocessor': preprocessor, 
            'labtrans': labtrans,
            'baseline_hazards': baseline_hazards
        }
        print("Cox-Time model loaded successfully.")
    except Exception as e:
        print(f"Warning: Could not load Cox-Time model. {e}")


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
        prediction_time = 30
        
        if model_name == 'keras_pca':
            artifacts = loaded_artifacts['keras_pca']
            keras_cols = artifacts['scaler'].feature_names_in_
            input_df_keras = input_df[keras_cols]
            
            scaled_features = artifacts['scaler'].transform(input_df_keras)
            pca_features = artifacts['pca'].transform(scaled_features)
            prediction_prob = artifacts['model'].predict(pca_features)[0][0]
            survival_probability = 1 - prediction_prob

        elif model_name == 'rsf':
            artifacts = loaded_artifacts['rsf']
            survival_funcs = artifacts['model'].predict_survival_function(input_df)
            survival_probability = survival_funcs[0](prediction_time) 

        elif model_name == 'deephit':
            artifacts = loaded_artifacts['deephit']
            processed_input = artifacts['preprocessor'].transform(input_df).astype('float32')
            surv_df = artifacts['model'].predict_surv_df(processed_input)
            
            idx = np.searchsorted(surv_df.columns, prediction_time, side='right') - 1
            survival_probability = surv_df.iloc[0, max(0, idx)]

        elif model_name == 'coxtime':
            # --- THIS IS THE FINAL, CORRECTED LOGIC ---
            artifacts = loaded_artifacts['coxtime']
            processed_input = artifacts['preprocessor'].transform(input_df).astype('float32')
            
            # 1. Attach the loaded baseline hazards to the model instance.
            #    The attribute name must end with an underscore.
            artifacts['model'].baseline_hazards_ = artifacts['baseline_hazards']
            
            # 2. Call predict_surv_df WITHOUT the keyword argument.
            #    The model will now find the .baseline_hazards_ attribute internally.
            surv_df = artifacts['model'].predict_surv_df(processed_input)
            
            # 3. Interpolate the result as before.
            survival_probability = np.interp(
                prediction_time,
                surv_df.index,
                surv_df.iloc[:, 0]
            )
            # ----------------------------------------------

        else:
            return jsonify({'error': 'Invalid model name specified.'}), 400

        return jsonify({'survival_probability': round(float(survival_probability) * 100, 2)})
        
    except Exception as e:
        print(f"Error during prediction for model '{model_name}': {type(e).__name__} - {str(e)}")
        return jsonify({'error': f'An error occurred during prediction: {str(e)}'}), 500

# --- Load all models when the Flask application starts ---
load_all_models()

if __name__ == '__main__':
    app.run(port=5001, debug=True)