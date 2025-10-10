from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd
import joblib
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__)
CORS(app)

# --- CHANGE: Loading the standard model artifacts ---
try:
    model_path = os.path.join(SCRIPT_DIR, 'survival_model.h5')
    model = load_model(model_path)
    
    scaler_path = os.path.join(SCRIPT_DIR, 'scaler.pkl')
    scaler = joblib.load(scaler_path)

    pca_path = os.path.join(SCRIPT_DIR, 'pca.pkl')
    pca = joblib.load(pca_path)

    print("Standard survival model, scaler, and PCA loaded successfully.")
except Exception as e:
    print(f"Error loading artifacts. Please run the training script first. Details: {e}")
    model, scaler, pca = None, None, None

@app.route('/predict_survival', methods=['POST'])
def predict():
    if not all([model, scaler, pca]):
        return jsonify({'error': 'Model/preprocessors not loaded. Check server logs.'}), 500

    data = request.get_json()
    if not data:
        return jsonify({'error': 'No input data provided'}), 400
        
    try:
        # The scaler expects only the 5 original features
        input_df = pd.DataFrame([data], columns=scaler.feature_names_in_)
        
        # Apply the full transformation pipeline
        scaled_features = scaler.transform(input_df)
        pca_features = pca.transform(scaled_features)
        
        # Make a prediction
        prediction_prob = model.predict(pca_features)[0][0]
        survival_probability = 1 - prediction_prob

        return jsonify({'survival_probability': round(float(survival_probability) * 100, 2)})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(port=5001, debug=True)