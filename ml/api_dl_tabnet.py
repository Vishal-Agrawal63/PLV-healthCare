# PATH: /ml/api_dl_tabnet.py (NEW PCA VERSION)

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
import joblib # <-- Import Joblib
from pytorch_tabnet.tab_model import TabNetClassifier
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__)
CORS(app)

# --- Load the saved models and pipeline ---
try:
    # Load the TabNet model trained on PCA data
    model_path = os.path.join(SCRIPT_DIR, 'survival_model_tabnet_pca.zip')
    tabnet_pca_model = TabNetClassifier()
    tabnet_pca_model.load_model(model_path)

    # Load the PCA pipeline
    pipeline_path = os.path.join(SCRIPT_DIR, 'pca_pipeline.pkl')
    pca_pipeline = joblib.load(pipeline_path)

    print("TabNet+PCA model and pipeline loaded successfully.")
except Exception as e:
    print(f"Error loading model or pipeline: {e}")
    tabnet_pca_model = None
    pca_pipeline = None

# --- NEW ENDPOINT FOR PCA PREDICTIONS ---
@app.route('/predict_survival_pca', methods=['POST'])
def predict_pca():
    if not tabnet_pca_model or not pca_pipeline:
        return jsonify({'error': 'TabNet+PCA model or pipeline not loaded'}), 500

    data = request.get_json()
    if not data:
        return jsonify({'error': 'No input data provided'}), 400
        
    try:
        input_df = pd.DataFrame([data])
        
        # Step 1: Transform the incoming data using the saved PCA pipeline
        transformed_features = pca_pipeline.transform(input_df).astype(np.float32)
        
        # Step 2: Make a prediction with TabNet on the transformed data
        prediction_prob = tabnet_pca_model.predict_proba(transformed_features)[0][1]
        
        survival_probability = 1 - prediction_prob

        return jsonify({'survival_probability': round(float(survival_probability) * 100, 2)})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # We still use port 5002 for the TabNet API server
    app.run(port=5002, debug=True)