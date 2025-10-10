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
scaler = None
model_columns = None


def load_models():
    """
    Loads all trained models, the scaler, and the model columns from disk.
    """
    global models, scaler, model_columns

    # --- CHANGE #1: EXPANDED THE LIST TO LOAD ALL 9 MODELS ---
    # This list now includes all the models saved by the training script.
    model_names = [
        'linearregression', 'ridge', 'lasso', 'elasticnet', 'tweedie',
        'randomforest', 'xgboost', 'lightgbm', 'lightgbm_tweedie'
    ]

    for name in model_names:
        try:
            model_path = os.path.join(SCRIPT_DIR, f'{name}.pkl')
            if os.path.exists(model_path):
                models[name] = joblib.load(model_path)
                print(f"Loaded model: {name}.pkl")
        except Exception as e:
            print(f"Warning: Could not load {name}.pkl. Error: {e}")

    # Load the scaler and model columns, which are essential for preprocessing
    try:
        scaler_path = os.path.join(SCRIPT_DIR, 'scaler.pkl')
        scaler = joblib.load(scaler_path)
        print("Scaler loaded successfully.")

        columns_path = os.path.join(SCRIPT_DIR, 'model_columns.pkl')
        model_columns = joblib.load(columns_path)
        print("Model columns loaded successfully.")
    except FileNotFoundError as e:
        print(f"Error: Could not find a required file. Please run the training script first. Details: {e}")


def engineer_features_api(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies the same feature engineering steps used during training to new data.
    """
    df = df.copy()
    # This list of engineered features must match the training script exactly.
    df['TotalVisits'] = df['OfficeVisits'] + df['OutpatientVisits'] + df['ERVisits']
    df['Has_HospitalStay'] = (df['HospitalDischarges'] > 0).astype(int)
    df['Age_x_HealthStatus'] = df['Age'] * df['HealthStatus']
    df['Log_TotalVisits'] = np.log1p(df['TotalVisits'])
    df['Visits_per_Age'] = df['TotalVisits'] / (df['Age'] + 1)
    df['Is_Senior'] = (df['Age'] >= 65).astype(int)
    df['Chronic_Risk'] = df['HealthStatus'] * df['TotalVisits']
    df['ERVisits_per_Age'] = df['ERVisits'] / (df['Age'] + 1)
    df['Outpatient_to_OfficeRatio'] = df['OutpatientVisits'] / (df['OfficeVisits'] + 1)
    df['Visits_x_HealthStatus'] = df['TotalVisits'] * df['HealthStatus']
    return df


# Load all assets when the application starts
load_models()


@app.route('/predict', methods=['POST'])
def predict():
    """
    Handles the prediction request from the frontend.
    """
    if not models or not scaler or not model_columns:
        return jsonify({'error': 'Models or preprocessing assets not loaded. Check server logs.'}), 500

    data = request.get_json()
    if not data or 'model_name' not in data or 'patient_data' not in data:
        return jsonify({'error': 'Invalid input: Missing model_name or patient_data.'}), 400

    model_name = data['model_name'].lower()
    patient_data = data['patient_data']

    selected_model = models.get(model_name)
    if not selected_model:
        return jsonify({'error': f"Model '{model_name}' not found. Available: {list(models.keys())}"}), 400

    try:
        # --- Preprocessing Pipeline ---
        # 1. Create DataFrame from input
        input_df = pd.DataFrame([patient_data])

        # 2. Engineer features
        featured_df = engineer_features_api(input_df)

        # 3. One-hot encode categorical features
        # Note: The categorical columns must match those from the training script
        categorical_cols = ['Sex', 'Race', 'PovertyCategory', 'InsuranceCoverage', 'HealthStatus']
        processed_df = pd.get_dummies(featured_df, columns=categorical_cols, drop_first=True)

        # 4. Align columns with the training set
        processed_df = processed_df.reindex(columns=model_columns, fill_value=0)

        # 5. Scale the features using the loaded scaler
        scaled_df = scaler.transform(processed_df)


        # --- CHANGE #2: CORRECTED PREDICTION LOGIC ---
        # This logic now correctly identifies which models were trained on the raw dollar
        # amount vs. the log-transformed amount.
        models_trained_on_dollars = [
            'randomforest', 'xgboost', 'lightgbm', 'lightgbm_tweedie', 'tweedie'
        ]

        if model_name in models_trained_on_dollars:
            # These models predict the final expenditure directly.
            prediction = selected_model.predict(scaled_df)
        else:
            # These models (Linear, Ridge, etc.) predict the log of the expenditure.
            # We must apply the inverse transformation (np.expm1) to get the dollar amount.
            prediction_log = selected_model.predict(scaled_df)
            prediction = np.expm1(prediction_log)

        final_prediction = float(prediction[0])

        return jsonify({'predicted_expenditure': round(final_prediction, 2)})

    except Exception as e:
        # Provide a more detailed error message for easier debugging.
        print(f"Error during prediction: {type(e).__name__} - {str(e)}")
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500


if __name__ == '__main__':
    # Use port 5000 for the ML API
    app.run(port=5000, debug=True)