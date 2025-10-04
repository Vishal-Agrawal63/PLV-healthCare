# PATH: /ml/train_dl.py
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def train_survival_model():
    print("Loading cleaned SUPPORT data...")
    input_path = os.path.join(SCRIPT_DIR, 'support_cleaned.csv')
    
    try:
        df = pd.read_csv(input_path)
    except FileNotFoundError:
        print(f"Error: '{input_path}' not found. Please run preprocess_dl.py first.")
        return

    # Define features (X) and target (y)
    X = df.drop('died', axis=1)
    y = df['died']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scale the features. This is very important for neural networks.
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # --- Define the Deep Learning Model ---
    model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        Dropout(0.3), # Dropout layer to prevent overfitting
        Dense(32, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid') # Sigmoid activation for a probability output (0 to 1)
    ])
    
    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    print("Training the deep learning model...")
    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=1)
    
    # Evaluate the model
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"\nModel Evaluation on Test Data: Loss={loss:.4f}, Accuracy={accuracy:.4f}")
    
    # Save the trained model
    model_path = os.path.join(SCRIPT_DIR, 'survival_model.h5')
    model.save(model_path)
    print(f"Deep learning model saved to {model_path}")

if __name__ == '__main__':
    train_survival_model()