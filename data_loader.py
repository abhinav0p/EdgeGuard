# FILE 2 — data_loader.py
# This script is responsible for loading the NASA CMAPSS FD001 dataset files.
# The files are space-separated, contain no headers, and represent time-series engine readings.

import os
import pandas as pd
import urllib.request
import urllib.error

# Define the dataset directory path
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "CMAPSSData")
TRAIN_FILE = os.path.join(DATA_DIR, "train_FD001.txt")
TEST_FILE = os.path.join(DATA_DIR, "test_FD001.txt")

# Define the correct column names for the CMAPSS dataset
# 1. engine_id (unit number)
# 2. cycle (time in cycles)
# 3. op1, op2, op3 (three operational settings)
# 4. s1 through s21 (21 sensor measurements)
COLUMNS = ["engine_id", "cycle", "op1", "op2", "op3"] + [f"s{i}" for i in range(1, 22)]

def download_cmapss_dataset():
    """
    Downloads the required files of the CMAPSS dataset (FD001) from a public repository
    if they are not already present in DATA_DIR.
    """
    files_to_download = ["train_FD001.txt", "test_FD001.txt", "RUL_FD001.txt"]
    base_url = "https://raw.githubusercontent.com/edwardzjl/CMAPSSData/master/"
    
    os.makedirs(DATA_DIR, exist_ok=True)
    
    for filename in files_to_download:
        file_path = os.path.join(DATA_DIR, filename)
        if not os.path.exists(file_path):
            url = base_url + filename
            print(f"Downloading {filename} from {url}...")
            try:
                urllib.request.urlretrieve(url, file_path)
                print(f"Successfully downloaded {filename}.")
            except urllib.error.URLError as e:
                print(f"Failed to download {filename}: {e}")
                raise FileNotFoundError(
                    f"Required dataset file {filename} could not be downloaded and does not exist locally. "
                    f"Error details: {e}"
                )

def load_dataset(file_path):
    """
    Loads a space-separated CMAPSS dataset file without header and assigns correct columns.
    Handles variable spacing between columns.
    Auto-downloads the dataset if not present.
    """
    if not os.path.exists(file_path):
        download_cmapss_dataset()
        
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Dataset file not found at: {os.path.abspath(file_path)}\n"
            f"Please download the CMAPSS dataset and place it in the 'data/CMAPSSData/' directory."
        )
    
    # Read space-separated file (sep=r'\s+' handles multiple spaces correctly)
    # header=None indicates there is no header row in the file
    df = pd.read_csv(file_path, sep=r"\s+", header=None, names=COLUMNS)
    return df

def main():
    print("=" * 60)
    print("EdgeGuard Data Loader Utility")
    print("=" * 60)
    
    # Try loading the training and test files
    try:
        print(f"Loading training data from: {TRAIN_FILE}...")
        train_df = load_dataset(TRAIN_FILE)
        print(f"Successfully loaded training data! Shape: {train_df.shape}")
        print("\nFirst 5 rows of Training Data:")
        print(train_df.head())
        print("-" * 60)
        
        print(f"Loading test data from: {TEST_FILE}...")
        test_df = load_dataset(TEST_FILE)
        print(f"Successfully loaded test data! Shape: {test_df.shape}")
        print("\nFirst 5 rows of Test Data:")
        print(test_df.head())
        print("=" * 60)
        
    except FileNotFoundError as e:
        print("\n[ERROR] Data Loading Failed:")
        print(e)
        print("\nTo download the dataset:")
        print("1. Go to: https://data.nasa.gov/Aerospace/CMAPSS-Jet-Engine-Simulated-Data/ff5v-h3wy")
        print("2. Download the zip folder containing CMAPSS dataset.")
        print("3. Extract and place train_FD001.txt, test_FD001.txt, and RUL_FD001.txt inside a folder named 'data/CMAPSSData/' relative to this script.")
        print("=" * 60)

if __name__ == "__main__":
    main()
