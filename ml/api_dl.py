# api_dl.py (Definitive Final Version)
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
import os
import torch
from torch import nn

from tensorflow.keras.models import load_model as load_keras_model
from pycox import models as pycox_models
from sksurv.ensemble import RandomSurvivalForest
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

app = Flask(__name__)
CORS(app)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
loaded_artifacts = {}

class CoxTimeMLP(nn.Module):
    # ... (this class is correct and remains unchanged) ...
    def __init__(self, in_features, hidden_dims=[64, 32]):
        super().__init__()
        all_dims = [in_features + 1] + hidden_dims + [1]
        layers = [];
        for i in range(len(all_dims) - 2):
            layers.extend([nn.Linear(all_dims[i], all_dims[i+1]), nn.ReLU(), nn.Dropout(0.3)])
        layers.append(nn.Linear(all_dims[-2], all_dims[-1]))
        self.network = nn.Sequential(*layers)
    def forward(self, x_features, x_time):
        x_combined = torch.cat([x_features, x_time], dim=1)
        return self.network(x_combined)

def load_all_models():
    """Load all PCA-based models and preprocessors."""
    try:
        scaler = joblib.load(os.path.join(SCRIPT_DIR, 'universal_pca_scaler.pkl'))
        pca = joblib.load(os.path.join(SCRIPT_DIR, 'universal_pca_transformer.pkl'))
        in_features_pca = pca.n_components_
        loaded_artifacts['preprocessors'] = {'scaler': scaler, 'pca': pca}
        print("Universal PCA preprocessors loaded.")
        print("SCALER EXPECTS:", scaler.feature_names_in_)

        # --- Load and prepare SORTED data for baseline calculation ---
        try:
            df_train_sample = pd.read_csv(os.path.join(SCRIPT_DIR, 'support_survival.csv')).sample(200, random_state=42)
            
            # =========================================================
            # --- FINAL, EXPLICIT FIX FOR SORTING ---
            # 1. Sort the entire sample DataFrame.
            df_baseline_sorted = df_train_sample.sort_values(by='d.time', ascending=False)
            
            # 2. Create the features and target variables directly from this sorted DataFrame.
            features_baseline = scaler.transform(df_baseline_sorted[scaler.feature_names_in_])
            pca_features_baseline = pca.transform(features_baseline).astype('float32')
            target_baseline = (df_baseline_sorted['d.time'].values, df_baseline_sorted['hospdead'].values)
            # =========================================================

            print("Loaded and sorted a sample of training data for baselines.")
        except Exception as e:
            print(f"CRITICAL WARNING: Could not load sample data. {e}")
            return

        # --- Load models ---
        # ... (Loading Keras, RSF, DeepHit remains unchanged)
        loaded_artifacts['keras_pca'] = {'model': load_keras_model(os.path.join(SCRIPT_DIR, 'keras_pca_model.h5'))}; print("Keras+PCA model loaded.")
        loaded_artifacts['rsf_pca'] = {'model': joblib.load(os.path.join(SCRIPT_DIR, 'rsf_pca_model.pkl'))}; print("RSF+PCA model loaded.")
        labtrans_dh = joblib.load(os.path.join(SCRIPT_DIR, 'deephit_pca_labtrans.pkl'))
        net_dh = nn.Sequential(nn.Linear(in_features_pca, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.3), nn.Linear(32, labtrans_dh.out_features))
        deephit_model = pycox_models.DeepHitSingle(net_dh, duration_index=labtrans_dh.cuts)
        deephit_model.load_model_weights(os.path.join(SCRIPT_DIR, 'deephit_pca_model.pt'))
        loaded_artifacts['deephit_pca'] = {'model': deephit_model, 'labtrans': labtrans_dh}; print("DeepHit+PCA model loaded.")
        
        # --- DeepSurv+PCA ---
        net_ds = nn.Sequential(nn.Linear(in_features_pca, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.3), nn.Linear(32, 1))
        deepsurv_model = pycox_models.CoxPH(net_ds)
        deepsurv_model.load_model_weights(os.path.join(SCRIPT_DIR, 'deepsurv_pca_model.pt'))
        deepsurv_model.compute_baseline_hazards(input=pca_features_baseline, target=target_baseline)
        loaded_artifacts['deepsurv_pca'] = {'model': deepsurv_model}; print("DeepSurv+PCA model loaded.")

        # --- Cox-Time+PCA ---
        labtrans_ct = joblib.load(os.path.join(SCRIPT_DIR, 'coxtime_pca_labtrans.pkl'))
        net_ct = CoxTimeMLP(in_features=in_features_pca)
        coxtime_model = pycox_models.CoxTime(net_ct, labtrans=labtrans_ct)
        coxtime_model.load_model_weights(os.path.join(SCRIPT_DIR, 'coxtime_pca_model.pt'))
        # --- Call with the same signature as DeepSurv, using the sorted variables ---
        coxtime_model.compute_baseline_hazards(input=pca_features_baseline, target=target_baseline)
        loaded_artifacts['coxtime_pca'] = {'model': coxtime_model}; print("Cox-Time+PCA model loaded.")

    except Exception as e:
        print(f"CRITICAL ERROR during model loading: {e}")
        import traceback; traceback.print_exc()


# --- The rest of the file (/predict_survival route) does NOT need to be changed ---
# ... (rest of the code is the same as the last working version) ...
@app.route('/predict_survival', methods=['POST'])
def predict():
    if 'preprocessors' not in loaded_artifacts: return jsonify({'error': 'Preprocessors not loaded.'}), 500
    data = request.get_json()
    if not data or 'model_name' not in data or 'patient_data' not in data: return jsonify({'error': 'Invalid payload.'}), 400
    model_name = data['model_name']
    if model_name not in loaded_artifacts: return jsonify({'error': f"Model '{model_name}' not loaded."}), 404
    try:
        scaler = loaded_artifacts['preprocessors']['scaler']
        pca = loaded_artifacts['preprocessors']['pca']
        patient_data = data['patient_data']
        input_df = pd.DataFrame([patient_data], columns=scaler.feature_names_in_)
        scaled_features = scaler.transform(input_df)
        pca_features = pca.transform(scaled_features).astype('float32')
        model = loaded_artifacts[model_name]['model']
        if model_name == 'keras_pca':
            prediction_prob = model.predict(pca_features, verbose=0)[0][0]
            survival_probability = 1 - prediction_prob
        elif model_name == 'rsf_pca':
            survival_funcs = model.predict_survival_function(pca_features)
            survival_probability = survival_funcs[0](30)
        elif model_name in ['deephit_pca', 'deepsurv_pca', 'coxtime_pca']:
            surv_df = model.predict_surv_df(pca_features)
            prediction_time = 30
            idx = np.searchsorted(surv_df.index if model_name != 'deephit_pca' else surv_df.columns, prediction_time, side='right') - 1
            survival_probability = surv_df.iloc[0, max(0, idx)]
        else: return jsonify({'error': 'Invalid model name.'}), 400
        return jsonify({'survival_probability': round(float(survival_probability) * 100, 2)})
    except Exception as e:
        print(f"Prediction Error for {model_name}: {e}")
        import traceback; traceback.print_exc()
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500

load_all_models()

if __name__ == '__main__':
    app.run(port=5001, debug=True)