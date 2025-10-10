import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def train_survival_model_with_pca():
    print("Loading original cleaned SUPPORT data...")
    # --- CHANGE: Using support_cleaned.csv ---
    input_path = os.path.join(SCRIPT_DIR, 'support_cleaned.csv')
    
    try:
        df = pd.read_csv(input_path)
    except FileNotFoundError:
        print(f"Error: '{input_path}' not found. Please ensure it is in the /ml directory.")
        return

    # Using the 5 original features from the file
    X = df.drop('died', axis=1)
    y = df['died']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"\nApplying PCA to the {X.shape[1]} original features...")
    pca = PCA(n_components=0.95)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    
    print(f"PCA reduced the number of features from {X.shape[1]} to {pca.n_components_}")
    
    model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train_pca.shape[1],)),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    print("\nTraining model on original PCA components...")
    model.fit(X_train_pca, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=1)
    
    loss, accuracy = model.evaluate(X_test_pca, y_test)
    print(f"\nModel Evaluation on Test Data: Loss={loss:.4f}, Accuracy={accuracy:.4f}")
    
    # --- CHANGE: Saving the standard model artifacts ---
    model.save(os.path.join(SCRIPT_DIR, 'survival_model.h5'))
    joblib.dump(scaler, os.path.join(SCRIPT_DIR, 'scaler.pkl'))
    joblib.dump(pca, os.path.join(SCRIPT_DIR, 'pca.pkl'))

    print("\nSuccessfully saved standard model, scaler, and PCA transformer.")

if __name__ == '__main__':
    train_survival_model_with_pca()