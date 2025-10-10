# PATH: /ml/train_model_full.py
import pandas as pd
import numpy as np
import os
import joblib
from time import time

# Model imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, TweedieRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# --- Configuration ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_ROUND = 4
RANDOM_STATE = 42

# ======================
# Helper Metrics
# ======================
def smape(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    nonzero = denom != 0
    out = np.zeros_like(denom)
    out[nonzero] = np.abs(y_pred[nonzero] - y_true[nonzero]) / denom[nonzero]
    return np.mean(out) * 100.0

def accuracy_within(y_true, y_pred, pct=0.2):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return np.mean(np.abs(y_pred - y_true) <= pct * y_true) * 100

def log1p_transform(y):
    return np.log1p(y)

def safe_inv_log1p(y_log, max_value=20):
    y_log = np.clip(y_log, a_min=None, a_max=max_value)
    return np.expm1(y_log)

# ======================
# Feature Engineering
# ======================
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
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

# ======================
# Data Loading / Cleaning
# ======================
def load_and_prepare_data(file_path: str) -> pd.DataFrame:
    print(f"Loading original data from: {file_path}")
    df = pd.read_csv(file_path)

    feature_map = {
        'DUPERSID': 'PersonID',
        'AGE19X': 'Age',
        'SEX': 'Sex',
        'RACETHX': 'Race',
        'POVCAT19': 'PovertyCategory',
        'INSCOV19': 'InsuranceCoverage',
        'RTHLTH53': 'HealthStatus',
        'OBTOTV19': 'OfficeVisits',
        'OPTOTV19': 'OutpatientVisits',
        'ERTOT19': 'ERVisits',
        'IPDIS19': 'HospitalDischarges',
        'TOTEXP19': 'TotalExpenditure',
    }
    df = df[list(feature_map.keys())].copy()
    df.rename(columns=feature_map, inplace=True)

    # Replace negative survey codes with NaN
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].apply(lambda x: np.nan if x < 0 else x)

    # Impute missing values
    for col in ['Age', 'OfficeVisits', 'OutpatientVisits', 'ERVisits', 'HospitalDischarges']:
        df[col].fillna(df[col].median(), inplace=True)
    for col in ['Sex', 'Race', 'PovertyCategory', 'InsuranceCoverage', 'HealthStatus']:
        df[col].fillna(df[col].mode()[0], inplace=True)

    # Drop rows with missing target
    df.dropna(subset=['TotalExpenditure'], inplace=True)

    # Cap target at 99th percentile
    cap_99 = df['TotalExpenditure'].quantile(0.99)
    df['TotalExpenditure'] = np.clip(df['TotalExpenditure'], a_min=0, a_max=cap_99)

    return df

# ======================
# Evaluation wrapper
# ======================
def evaluate_and_store_results(name, model, X_test_scaled, y_test_dollars, results_dict, train_seconds, use_log=True):
    y_pred = model.predict(X_test_scaled)
    if use_log:
        y_pred_dollars = safe_inv_log1p(y_pred)
    else:
        y_pred_dollars = y_pred

    r2 = r2_score(y_test_dollars, y_pred_dollars)
    mae = mean_absolute_error(y_test_dollars, y_pred_dollars)
    rmse = mean_squared_error(y_test_dollars, y_pred_dollars, squared=False)
    s_mape = smape(y_test_dollars, y_pred_dollars)
    acc_20 = accuracy_within(y_test_dollars, y_pred_dollars, pct=0.2)

    results_dict[name] = {
        'R-squared': r2,
        'MAE': mae,
        'RMSE': rmse,
        'SMAPE (%)': s_mape,
        'Accuracy ±20%': acc_20,
        'Training Time (s)': train_seconds,
    }

# ======================
# Main training pipeline
# ======================
def train_and_evaluate_models():
    h216_path = os.path.join(SCRIPT_DIR, 'h216.csv')
    if not os.path.exists(h216_path):
        print("Error: 'h216.csv' not found in the /ml directory.")
        return

    df = load_and_prepare_data(h216_path)
    df_featured = engineer_features(df)

    # One-hot encode categorical features
    categorical_cols = ['Sex', 'Race', 'PovertyCategory', 'InsuranceCoverage', 'HealthStatus']
    df_enc = pd.get_dummies(df_featured, columns=categorical_cols, drop_first=True)

    # Prepare X, y
    X = df_enc.drop(columns=['TotalExpenditure', 'PersonID'])
    y_dollars = df_enc['TotalExpenditure']
    y_log = log1p_transform(y_dollars)

    # Stratified split
    bins = pd.qcut(y_dollars, q=10, labels=False, duplicates='drop')
    X_train, X_test, y_train_log, y_test_log, bins_train, bins_test = train_test_split(
        X, y_log, bins, test_size=0.2, random_state=RANDOM_STATE, stratify=bins
    )
    y_test_dollars = safe_inv_log1p(y_test_log)

    # StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler, os.path.join(SCRIPT_DIR, 'scaler.pkl'))
    joblib.dump(X.columns.tolist(), os.path.join(SCRIPT_DIR, 'model_columns.pkl'))

    # ======================
    # Define models
    # ======================
    models = {
        # Linear models with log1p
        'LinearRegression': LinearRegression(),
        'Ridge': Ridge(alpha=1.0),
        'Lasso': Lasso(alpha=0.1),
        'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5),
        'Tweedie': TweedieRegressor(power=1.5, alpha=0.0, link='log', max_iter=1000),

        # Tree-based models (no log transform)
        'RandomForest': RandomForestRegressor(
            n_estimators=600, max_depth=None, min_samples_split=4, min_samples_leaf=2,
            n_jobs=-1, random_state=RANDOM_STATE
        ),
        'XGBoost': XGBRegressor(
            n_estimators=900, learning_rate=0.05, max_depth=6, subsample=0.9,
            colsample_bytree=0.9, reg_alpha=0.0, reg_lambda=1.0,
            min_child_weight=1.0, n_jobs=-1, random_state=RANDOM_STATE, tree_method='hist'
        ),
        'LightGBM': LGBMRegressor(
            n_estimators=1000, learning_rate=0.05, num_leaves=63, max_depth=-1,
            min_child_samples=20, subsample=0.9, colsample_bytree=0.9,
            reg_alpha=0.0, reg_lambda=0.0, n_jobs=-1, random_state=RANDOM_STATE, verbose=-1
        ),
        'LightGBM_Tweedie': LGBMRegressor(
            objective='tweedie', tweedie_variance_power=1.5,
            n_estimators=1000, learning_rate=0.05, num_leaves=63, max_depth=-1,
            min_child_samples=20, subsample=0.9, colsample_bytree=0.9,
            reg_alpha=0.0, reg_lambda=0.0, n_jobs=-1, random_state=RANDOM_STATE, verbose=-1
        )
    }

    results = {}

    # Train and evaluate
    for name, model in models.items():
        print(f"\n--- Training {name} ---")
        t0 = time()
        if name in ['RandomForest', 'XGBoost', 'LightGBM', 'LightGBM_Tweedie']:
            model.fit(X_train_scaled, safe_inv_log1p(y_train_log))
            use_log = False
        else:
            model.fit(X_train_scaled, y_train_log)
            use_log = True
        train_time = round(time() - t0, 2)
        evaluate_and_store_results(name, model, X_test_scaled, y_test_dollars, results, train_time, use_log=use_log)
        joblib.dump(model, os.path.join(SCRIPT_DIR, f"{name.lower()}.pkl"))
        print(f"Saved model to '{name.lower()}.pkl'")

    # Ensemble (average of tree predictions)
    y_pred_ensemble = (
        models['RandomForest'].predict(X_test_scaled) +
        models['XGBoost'].predict(X_test_scaled) +
        models['LightGBM'].predict(X_test_scaled)
    ) / 3
    results['Ensemble'] = {
        'R-squared': r2_score(y_test_dollars, y_pred_ensemble),
        'MAE': mean_absolute_error(y_test_dollars, y_pred_ensemble),
        'RMSE': mean_squared_error(y_test_dollars, y_pred_ensemble, squared=False),
        'SMAPE (%)': smape(y_test_dollars, y_pred_ensemble),
        'Accuracy ±20%': accuracy_within(y_test_dollars, y_pred_ensemble, pct=0.2),
        'Training Time (s)': np.nan
    }

    # Display results
    results_df = pd.DataFrame(results).T
    display_df = results_df.copy()
    display_df['MAE'] = display_df['MAE'].map(lambda x: f"${x:,.2f}")
    display_df['RMSE'] = display_df['RMSE'].map(lambda x: f"${x:,.2f}")
    display_df['R-squared'] = display_df['R-squared'].map(lambda v: f"{v:.{RESULTS_ROUND}f}")
    display_df['SMAPE (%)'] = display_df['SMAPE (%)'].map(lambda v: f"{v:.2f}%")
    display_df['Accuracy ±20%'] = display_df['Accuracy ±20%'].map(lambda v: f"{v:.2f}%")
    columns = ['R-squared', 'MAE', 'RMSE', 'SMAPE (%)', 'Accuracy ±20%', 'Training Time (s)']
    display_df = display_df[columns]

    try:
        print(display_df.sort_values(by='MAE', ascending=True))
    except Exception:
        print(display_df)

    print("--------------------------------")
    print("Training complete.")

if __name__ == '__main__':
    train_and_evaluate_models()
