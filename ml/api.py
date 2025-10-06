# PATH: /ml/api.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__)
CORS(app)
models = {}
model_columns = None


def load_models():
    global models, model_columns
    model_names = ['randomforest', 'xgboost', 'lightgbm', 'lightgbm_tweedie']
    for name in model_names:
        try:
            model_path = os.path.join(SCRIPT_DIR, f'{name}.pkl')
            if os.path.exists(model_path):
                models[name] = joblib.load(model_path)
                print(f"Loaded model: {name}.pkl")
        except Exception as e:
            print(f"Warning: Could not load {name}.pkl. Error: {e}")
    try:
        columns_path = os.path.join(SCRIPT_DIR, 'model_columns.pkl')
        model_columns = joblib.load(columns_path)
        print("Model columns loaded successfully.")
    except FileNotFoundError:
        print("Error: model_columns.pkl not found. Train models first.")


def engineer_features_api(df: pd.DataFrame) -> pd.DataFrame:
    # This function is correct and does not need changes
    df = df.copy()
    df['TotalVisits'] = df['OfficeVisits'] + df['OutpatientVisits'] + df['ERVisits']
    df['Has_HospitalStay'] = (df['HospitalDischarges'] > 0).astype(int)
    df['Age_x_HealthStatus'] = df['Age'] * df['HealthStatus']
    df['Log_TotalVisits'] = np.log1p(df['TotalVisits'])
    df['Visits_per_Age'] = df['TotalVisits'] / (df['Age'] + 1)
    df['Is_Senior'] = (df['Age'] >= 65).astype(int)
    df['Chronic_Risk'] = df['HealthStatus'] * df['TotalVisits']
    return df


load_models()


@app.route('/predict', methods=['POST'])
def predict():
    if not models or model_columns is None:
        return jsonify({'error': 'Models not loaded. Check server logs.'}), 500

    data = request.get_json()
    if not data or 'model_name' not in data or 'patient_data' not in data:
        return jsonify({'error': 'Invalid input format.'}), 400

    model_name = data['model_name'].lower()
    
    # --- THIS IS THE FIX ---
    # We must extract the patient_data dictionary before we can use it.
    patient_data = data['patient_data']
    
    selected_model = models.get(model_name)
    if not selected_model:
        return jsonify({'error': f"Model '{model_name}' not found. Available models: {list(models.keys())}"}), 400

    try:
        # Now the 'patient_data' variable exists and this line will work
        input_df = pd.DataFrame([patient_data])
        
        featured_df = engineer_features_api(input_df)
        processed_df = pd.get_dummies(featured_df, columns=['Sex', 'Race', 'PovertyCategory', 'InsuranceCoverage', 'HealthStatus'], drop_first=True)
        processed_df = processed_df.reindex(columns=model_columns, fill_value=0)

        if 'tweedie' in model_name:
            prediction_dollar = selected_model.predict(processed_df)
        else:
            prediction_log = selected_model.predict(processed_df)
            prediction_dollar = np.expm1(prediction_log)

        final_prediction = float(prediction_dollar[0])

        return jsonify({'predicted_expenditure': round(final_prediction, 2)})

    except Exception as e:
        # Add the exception type to the error for better debugging
        return jsonify({'error': f'An error occurred during prediction: {type(e).__name__} - {str(e)}'}), 500


if __name__ == '__main__':
    app.run(port=5000, debug=True)