# FILE 6 — fault_classifier.py
# This script trains a Random Forest Classifier to categorize the severity of developing faults.
# The categories are:
#   - NORMAL (0)
#   - HIGH DEGRADATION (1)
#   - CRITICAL (2)
# It evaluates the model on the test set using accuracy score and classification report.

import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from preprocess import preprocess_pipeline, MODELS_DIR

# Define model path
MODEL_PATH = os.path.join(MODELS_DIR, "fault_classifier.pkl")

def train_fault_classifier():
    print("=" * 60)
    print("EdgeGuard Fault Severity Classifier (Model 3) Training")
    print("=" * 60)
    
    # 1. Load preprocessed data
    print("Loading preprocessed datasets...")
    X_train, X_test, _, _, y_fault_train, y_fault_test = preprocess_pipeline()
    
    print(f"X_train shape: {X_train.shape} | X_test shape: {X_test.shape}")
    print(f"y_fault_train shape: {y_fault_train.shape} | y_fault_test shape: {y_fault_test.shape}")
    
    # 2. Train Random Forest Classifier
    print("\nInitializing Random Forest Classifier...")
    model = RandomForestClassifier(
        n_estimators=100, 
        random_state=42, 
        n_jobs=-1
    )
    
    print("Training Random Forest Classifier...")
    model.fit(X_train, y_fault_train)
    
    # 3. Evaluate on test set
    print("\nEvaluating on test set...")
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_fault_test, y_pred)
    print(f"Classification Accuracy: {acc * 100:.2f}%")
    
    print("\nClassification Report:")
    target_names = ["NORMAL (0)", "HIGH DEGRADATION (1)", "CRITICAL (2)"]
    print(classification_report(y_fault_test, y_pred, target_names=target_names))
    
    # 4. Save model
    print(f"\nSaving fault classifier model to: {MODEL_PATH}...")
    joblib.dump(model, MODEL_PATH)
    print("Fault classifier model successfully saved!")
    print("=" * 60)

if __name__ == "__main__":
    train_fault_classifier()
