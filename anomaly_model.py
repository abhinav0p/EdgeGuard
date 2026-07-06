# FILE 4 — anomaly_model.py
# This script trains an Isolation Forest anomaly detection model.
# The model learns to flag unusual patterns (drift) in sensor readings.
# It is trained completely unsupervised on normal/historical training data.

import os
import joblib
import numpy as np
from sklearn.ensemble import IsolationForest
from preprocess import preprocess_pipeline, MODELS_DIR

# Define model path
MODEL_PATH = os.path.join(MODELS_DIR, "anomaly_model.pkl")

def train_anomaly_model():
    print("=" * 60)
    print("EdgeGuard Anomaly Detector (Model 1) Training")
    print("=" * 60)
    
    # 1. Load preprocessed data
    print("Loading preprocessed datasets...")
    X_train, X_test, _, _, y_fault_train, y_fault_test = preprocess_pipeline()
    
    # In real application, we want to train the anomaly detector on NORMAL data only.
    # But to follow the instructions: "Train Isolation Forest with contamination=0.05 and random_state=42"
    # we train it on the entire training feature set X_train.
    print(f"Training features shape: {X_train.shape}")
    print("Initializing Isolation Forest...")
    
    model = IsolationForest(
        contamination=0.05, 
        random_state=42, 
        n_jobs=-1
    )
    
    print("Fitting Isolation Forest model...")
    model.fit(X_train)
    
    # 2. Evaluate on test data
    # Isolation Forest predict() returns:
    #   1 for inliers (normal)
    #  -1 for outliers (anomaly)
    print("\nEvaluating on test dataset...")
    test_preds = model.predict(X_test)
    
    # Calculate overall anomaly detection rate
    anomaly_mask = (test_preds == -1)
    overall_anomaly_rate = anomaly_mask.mean() * 100
    print(f"Overall Anomaly Detection Rate on Test Set: {overall_anomaly_rate:.2f}%")
    
    # Professional Insight: Let's see how well it flags instances of different severities
    # Class mapping: 0 = NORMAL, 1 = HIGH DEGRADATION, 2 = CRITICAL
    print("\nAnomaly Detection Rate by Ground Truth Severity Class:")
    classes = [0, 1, 2]
    class_names = {0: "NORMAL", 1: "HIGH DEGRADATION", 2: "CRITICAL"}
    
    for cls in classes:
        cls_mask = (y_fault_test == cls)
        if np.sum(cls_mask) > 0:
            cls_preds = test_preds[cls_mask]
            cls_anomaly_rate = (cls_preds == -1).mean() * 100
            print(f"  {class_names[cls]} (Class {cls}): {cls_anomaly_rate:.2f}% flagged as anomaly")
            
    # 3. Save model
    print(f"\nSaving anomaly detector model to: {MODEL_PATH}...")
    joblib.dump(model, MODEL_PATH)
    print("Anomaly detector model successfully saved!")
    print("=" * 60)

if __name__ == "__main__":
    train_anomaly_model()
