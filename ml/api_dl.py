# PATH: /ml/api_dl.py
from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd # Import pandas for one-hot encoding
from sklearn.preprocessing import StandardScaler
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__)
CORS(app)

# Load the trained model and pre-fit the scaler
try:
    model_path = os.path.join(SCRIPT_DIR, 'survival_model.h5')
    model = load_model(model_path)
    
    # We need to re-fit the scaler on the original data to use it for new predictions
    df_train = pd.read_csv(os.path.join(SCRIPT_DIR, 'support_cleaned.csv')).drop('died', axis=1)
    scaler = StandardScaler().fit(df_train)

    print("Survival DL model and scaler loaded successfully.")
except Exception as e:
    print(f"Error loading model or scaler: {e}")
    model = None
    scaler = None

@app.route('/predict_survival', methods=['POST'])
def predict():
    if not model or not scaler:
        return jsonify({'error': 'Model or scaler not loaded'}), 500

    data = request.get_json()
    if not data:
        return jsonify({'error': 'No input data provided'}), 400
        
    try:
        # Create a DataFrame from the input
        input_df = pd.DataFrame([data])
        
        # Scale the new data using the pre-fitted scaler
        scaled_features = scaler.transform(input_df)
        
        # Make a prediction
        prediction_prob = model.predict(scaled_features)[0][0]
        
        # Convert prediction to survival probability
        survival_probability = 1 - prediction_prob

        return jsonify({'survival_probability': round(float(survival_probability) * 100, 2)})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Run on a different port to avoid conflict with the first API
    app.run(port=5001, debug=True)