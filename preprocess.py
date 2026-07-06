# FILE 3 — preprocess.py
# This script preprocesses the raw engine telemetry data.
# It drops near-constant sensors, calculates Remaining Useful Life (RUL),
# normalizes the remaining sensors, saves the scaler, and generates fault severity labels.

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from data_loader import load_dataset, TRAIN_FILE, TEST_FILE, DATA_DIR

# Define directory to save models and scaler
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")

# Constant sensors to be dropped as they don't provide useful variance
DROPPED_SENSORS = ["s1", "s5", "s6", "s10", "s16", "s18", "s19"]

def calculate_train_rul(df):
    """
    Calculates Remaining Useful Life (RUL) for the training set.
    Since training engines are run to failure, RUL = max_cycle - current_cycle.
    """
    # Group by engine_id to find the maximum cycle for each engine
    max_cycle = df.groupby("engine_id")["cycle"].max().reset_index()
    max_cycle.rename(columns={"cycle": "max_cycle"}, inplace=True)
    
    # Merge max cycle info back into main dataframe
    df = df.merge(max_cycle, on="engine_id")
    
    # Calculate RUL
    df["RUL"] = df["max_cycle"] - df["cycle"]
    
    # Drop intermediate column
    df.drop("max_cycle", axis=1, inplace=True)
    return df

def calculate_test_rul(df):
    """
    Calculates Remaining Useful Life (RUL) for the test set.
    Since test engines are stopped before failure, we load the actual remaining RUL
    from RUL_FD001.txt and add it to the cycle difference.
    RUL_t = RUL_final + (max_cycle - t)
    """
    rul_file_path = os.path.join(DATA_DIR, "RUL_FD001.txt")
    if not os.path.exists(rul_file_path):
        raise FileNotFoundError(f"Test RUL ground truth file not found at: {rul_file_path}")
        
    # Read the ground truth RUL for each engine at its final test cycle
    test_rul_truth = pd.read_csv(rul_file_path, header=None, names=["final_rul"])
    # engine_id starts from 1, mapping to index of the file
    test_rul_truth["engine_id"] = test_rul_truth.index + 1
    
    # Find the maximum cycle observed for each test engine
    max_cycle = df.groupby("engine_id")["cycle"].max().reset_index()
    max_cycle.rename(columns={"cycle": "max_cycle"}, inplace=True)
    
    # Merge max cycle and final ground truth RUL
    test_info = max_cycle.merge(test_rul_truth, on="engine_id")
    
    # Merge back into test dataframe
    df = df.merge(test_info, on="engine_id")
    
    # Calculate current RUL: final_rul + (max_cycle - current_cycle)
    df["RUL"] = df["final_rul"] + (df["max_cycle"] - df["cycle"])
    
    # Drop intermediate columns
    df.drop(["max_cycle", "final_rul"], axis=1, inplace=True)
    return df

def get_fault_severity(rul):
    """
    Assigns fault severity label based on RUL:
    - NORMAL (0): RUL > 80
    - HIGH DEGRADATION (1): 30 <= RUL <= 80
    - CRITICAL (2): RUL < 30
    """
    if rul > 80:
        return 0
    elif rul >= 30:
        return 1
    else:
        return 2

def preprocess_pipeline():
    """
    Full pipeline to load, preprocess, scale, and label the dataset.
    Saves the trained scaler to models/scaler.pkl.
    """
    # Create models directory if it doesn't exist
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    # 1. Load data
    train_df = load_dataset(TRAIN_FILE)
    test_df = load_dataset(TEST_FILE)
    
    # 2. Calculate RUL and cap it at 125 to avoid outlier influence
    train_df = calculate_train_rul(train_df)
    test_df = calculate_test_rul(test_df)
    
    train_df["RUL_capped"] = train_df["RUL"].clip(upper=125)
    test_df["RUL_capped"] = test_df["RUL"].clip(upper=125)
    
    # 3. Create fault severity labels
    y_fault_train = train_df["RUL"].apply(get_fault_severity).values
    y_fault_test = test_df["RUL"].apply(get_fault_severity).values
    
    # Separate targets
    y_rul_train = train_df["RUL_capped"].values
    y_rul_test = test_df["RUL_capped"].values
    
    # 4. Identify remaining sensor columns
    all_sensors = [f"s{i}" for i in range(1, 22)]
    remaining_sensors = [s for s in all_sensors if s not in DROPPED_SENSORS]
    
    # Extract features (using only remaining sensor columns)
    X_train_raw = train_df[remaining_sensors].copy()
    X_test_raw = test_df[remaining_sensors].copy()
    
    # 5. Normalize remaining sensors with MinMaxScaler
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_test_scaled = scaler.transform(X_test_raw)
    
    # 6. Save scaler
    joblib.dump(scaler, SCALER_PATH)
    print(f"Scaler successfully fit and saved to: {SCALER_PATH}")
    
    # Convert scaled features back to DataFrame to preserve column names if needed,
    # or return as NumPy arrays. Let's return as NumPy arrays for model training.
    return X_train_scaled, X_test_scaled, y_rul_train, y_rul_test, y_fault_train, y_fault_test

def main():
    print("=" * 60)
    print("EdgeGuard Data Preprocessing Pipeline")
    print("=" * 60)
    
    try:
        X_train, X_test, y_rul_train, y_rul_test, y_fault_train, y_fault_test = preprocess_pipeline()
        
        print("\nPreprocessing completed successfully!")
        print(f"X_train shape: {X_train.shape} | X_test shape: {X_test.shape}")
        print(f"y_rul_train shape: {y_rul_train.shape} | y_rul_test shape: {y_rul_test.shape}")
        print(f"y_fault_train shape: {y_fault_train.shape} | y_fault_test shape: {y_fault_test.shape}")
        
        # Display label distribution
        print("\nTraining Fault Severity class distribution:")
        classes, counts = np.unique(y_fault_train, return_counts=True)
        for cls, count in zip(classes, counts):
            name = "NORMAL (0)" if cls == 0 else ("HIGH DEGRADATION (1)" if cls == 1 else "CRITICAL (2)")
            print(f"  {name}: {count} rows ({count/len(y_fault_train)*100:.1f}%)")
            
        print("\nTest Fault Severity class distribution:")
        classes_t, counts_t = np.unique(y_fault_test, return_counts=True)
        for cls, count in zip(classes_t, counts_t):
            name = "NORMAL (0)" if cls == 0 else ("HIGH DEGRADATION (1)" if cls == 1 else "CRITICAL (2)")
            print(f"  {name}: {count} rows ({count/len(y_fault_test)*100:.1f}%)")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n[ERROR] Preprocessing failed: {e}")
        print("=" * 60)

if __name__ == "__main__":
    main()
