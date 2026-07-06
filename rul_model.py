# FILE 5 — rul_model.py
# This script trains a Random Forest Regressor to predict the Remaining Useful Life (RUL)
# of an engine in cycles.
# It evaluates the model on the test set using Root Mean Squared Error (RMSE) and Mean Absolute Error (MAE).

import os
import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from preprocess import preprocess_pipeline, MODELS_DIR

# Define model path
MODEL_PATH = os.path.join(MODELS_DIR, "rul_model.pkl")

def train_rul_model():
    print("=" * 60)
    print("EdgeGuard RUL Regressor (Model 2) Training")
    print("=" * 60)
    
    # 1. Load preprocessed data
    print("Loading preprocessed datasets...")
    X_train, X_test, y_rul_train, y_rul_test, _, _ = preprocess_pipeline()
    
    print(f"X_train shape: {X_train.shape} | X_test shape: {X_test.shape}")
    print(f"y_rul_train shape: {y_rul_train.shape} | y_rul_test shape: {y_rul_test.shape}")
    
    # 2. Train Random Forest Regressor
    print("\nInitializing Random Forest Regressor...")
    model = RandomForestRegressor(
        n_estimators=100, 
        random_state=42, 
        n_jobs=-1
    )
    
    print("Training Random Forest Regressor (this may take a moment)...")
    model.fit(X_train, y_rul_train)
    
    # 3. Evaluate on test set
    print("\nEvaluating on test set...")
    y_pred = model.predict(X_test)
    
    rmse = np.sqrt(mean_squared_error(y_rul_test, y_pred))
    mae = mean_absolute_error(y_rul_test, y_pred)
    
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f} cycles")
    print(f"Mean Absolute Error (MAE): {mae:.4f} cycles")
    
    # Show example predictions vs ground truth
    print("\nSample Predictions vs Ground Truth (first 10 test samples):")
    for i in range(10):
        print(f"  Sample {i+1}: Predicted RUL = {y_pred[i]:.1f} | Actual RUL = {y_rul_test[i]:.1f}")
        
    # 4. Save model
    print(f"\nSaving RUL regressor model to: {MODEL_PATH}...")
    joblib.dump(model, MODEL_PATH)
    print("RUL regressor model successfully saved!")
    print("=" * 60)

if __name__ == "__main__":
    train_rul_model()
