# Technical Deep Dive: Patient Lifetime Value (PLV) Prediction Project

This document explains the technical implementation, machine learning methodology, and architectural decisions behind the PLV Prediction application.

## 1. Project Goal & Problem Definition

The primary goal of this project is to predict a patient's total annual healthcare expenditure based on a set of demographic, socioeconomic, and health-related characteristics.

This is fundamentally a **supervised regression problem**. We are given a dataset with labeled examples (patients with known expenditures) and our task is to train a model that can learn the relationship between a patient's features and their total spending.

---

## 2. The Machine Learning Core

The heart of this application is the predictive model. Here’s a breakdown of its development process.

### 2.1. The Dataset

*   **Source:** The model is trained on the **MEPS HC-216 (2019 Full-Year Consolidated Data)** dataset from the Medical Expenditure Panel Survey.
*   **Nature:** This is a public, high-quality dataset based on real-world surveys of individuals, providing a comprehensive view of their healthcare utilization and costs in the United States.

### 2.2. Target Variable (Dependent Variable)

The variable we aim to predict is the **dependent variable** or **target variable**.

*   **Variable Name:** `TOTEXP19`
*   **Description:** This column represents the **Total Healthcare Expenditures** for an individual for the entire year of 2019. It includes all payments from all sources (private insurance, Medicare, Medicaid, out-of-pocket, etc.).
*   **Role in Project:** This value serves as a powerful, single-year proxy for Patient Lifetime Value (PLV). A model that accurately predicts this annual cost can be foundational for longer-term financial forecasting.

### 2.3. Feature Selection (Independent Variables)

The variables used to make the prediction are the **independent variables** or **features**. The following key features were selected from the dataset based on their known influence on healthcare costs:

*   **`AGE19X` (Age):** A primary driver of health status and cost.
*   **`SEX` (Sex):** Biological sex, influences risk for certain conditions.
*   **`RACETHX` (Race/Ethnicity):** Correlates with socioeconomic and health factors.
*   **`POVCAT19` (Poverty Category):** Income level is a strong predictor of health and access to care.
*   **`INSCOV19` (Insurance Coverage):** Insurance status dictates how and when care is sought.
*   **`RTHLTH53` (Health Status):** Self-reported health is a robust predictor of future needs.
*   **`OBTOTV19` (Office Visits):** A measure of healthcare utilization frequency.
*   **`OPTOTV19` (Outpatient Visits):** Another key utilization metric.
*   **`ERTOT19` (ER Visits):** Indicates acute or severe health events.
*   **`IPDIS19` (Hospital Discharges):** Represents significant, high-cost health events.

### 2.4. Data Preprocessing (`preprocess.py`)

Raw data is never clean enough for modeling. The `preprocess.py` script performs the following critical steps:
1.  **Handling Missing Values:** In the MEPS dataset, negative values often signify "Don't Know," "Refused," or "Not Ascertained." These are treated as missing data (`NaN`).
2.  **Imputation:** Missing numerical values (like Age or Visits) are filled with the **median** of the column, which is robust to outliers. Missing categorical values are filled with the **mode** (most frequent value).
3.  **Categorical Encoding:** The machine learning model cannot process text-based or non-ordinal categories directly. We use **One-Hot Encoding** (via `pandas.get_dummies`) to convert variables like `Sex`, `Race`, etc., into a numerical format the model can understand. This creates binary (0/1) columns for each category.

### 2.5. Machine Learning Algorithm

*   **Algorithm Used:** **Random Forest Regressor** (`sklearn.ensemble.RandomForestRegressor`).

*   **Why this algorithm was chosen:**
    1.  **High Performance on Tabular Data:** Random Forests are consistently one of the best-performing "out-of-the-box" algorithms for structured/tabular data like ours.
    2.  **Robustness:** It is less sensitive to outliers and noisy data compared to simpler models.
    3.  **Non-Linearity:** It can capture complex, non-linear relationships between features and the target variable (e.g., the effect of age on cost is not a straight line).
    4.  **Reduced Overfitting:** By averaging the predictions of many individual decision trees, it significantly reduces the risk of overfitting, which is common in single decision tree models.

### 2.6. Model Training & Evaluation (`train_model.py`)

1.  **Splitting Data:** The preprocessed data is split into a **training set (80%)** and a **testing set (20%)**. The model learns *only* from the training set.
2.  **Training:** The `RandomForestRegressor` model is fitted on the training data.
3.  **Evaluation:** The model's performance is evaluated on the unseen testing set using standard regression metrics:
    *   **R-squared (R²):** Measures the proportion of the variance in the target variable that is predictable from the independent variables. Closer to 1 is better.
    *   **Mean Absolute Error (MAE):** The average absolute difference between the predicted and actual expenditures. This gives a clear, interpretable dollar amount of the average prediction error.
    *   **Root Mean Squared Error (RMSE):** Similar to MAE but penalizes larger errors more heavily.

### 2.7. Model Persistence & Serving (`api.py`)

1.  **Saving the Model:** After training, the model object is serialized and saved to a file named `model.pkl` using `joblib`. This captures the entire trained state of the model.
2.  **Serving the Model:** The `api.py` script uses **Flask** to create a lightweight web server. When this server starts, it loads the `model.pkl` file into memory. It exposes a `/predict` endpoint that:
    *   Accepts new patient data as JSON.
    *   Performs the same one-hot encoding on the new data to match the training format.
    *   Uses the loaded model to make a prediction.
    *   Returns the prediction as a JSON response.

---

## 3. Application Architecture

The project is intentionally decoupled into three distinct services to ensure separation of concerns and maintainability.

1.  **ML Service (Python/Flask):** Its *only* job is to make predictions. It knows nothing about databases or user interfaces.
2.  **Backend (Node.js/Express):** Acts as the central orchestrator. It handles all client-facing API requests, manages business logic, and communicates with both the database and the ML service. This separation means the frontend doesn't need to know where the ML model lives.
3.  **Frontend (React):** Its *only* job is to provide a user interface. It collects user input and displays data by communicating exclusively with the Node.js backend.

This three-tiered architecture is a robust and scalable pattern for building full-stack applications with a machine learning component.