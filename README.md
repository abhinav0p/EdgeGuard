# EdgeGuard: Edge AI System for Vehicle Health Monitoring and Predictive Maintenance

EdgeGuard is a lightweight on-device AI system designed to read vehicle sensor data continuously and predict component failures before they occur. It operates entirely offline without internet connection or cloud dependencies, allowing it to run efficiently on low-cost edge platforms like a Raspberry Pi 4 (under ₹5,000). By monitoring subtle sensor drift patterns over time, EdgeGuard aims to predict issues days or weeks in advance before the ECU trigger limits are ever crossed.

This prototype simulates real-time vehicle diagnostics using the NASA CMAPSS FD001 dataset, tracking simulated turbofan engines from a healthy state until failure.

---

## Tech Stack
*   **Data Processing & Machine Learning**: Python, Pandas, NumPy, Scikit-Learn, Joblib
*   **Real-time Visualization Dashboard**: Streamlit, Plotly, HTML/CSS
*   **Target Edge Platform**: Raspberry Pi 4 Model B (under ₹5,000, runs offline)

---

## Folder Structure
```text
EdgeGuard/
├── data/
│   └── CMAPSSData/
│       ├── train_FD001.txt
│       ├── test_FD001.txt
│       └── RUL_FD001.txt
├── models/
│   ├── scaler.pkl
│   ├── anomaly_model.pkl
│   ├── rul_model.pkl
│   └── fault_classifier.pkl
├── data_loader.py
├── preprocess.py
├── anomaly_model.py
├── rul_model.py
├── fault_classifier.py
├── app.py
├── requirements.txt
└── README.md
```

---

## Installation & Setup Instructions

### 1. Install Dependencies
Make sure you have Python 3.8+ installed, then run:
```bash
pip install -r requirements.txt
```

### 2. Download the NASA CMAPSS Dataset
EdgeGuard requires the NASA CMAPSS jet engine simulated data.
1. Download the dataset zip file from the [official NASA repository link](https://data.nasa.gov/docs/legacy/CMAPSSData.zip) or the stable Amazon S3 mirror:
   * `https://data.nasa.gov/docs/legacy/CMAPSSData.zip`
2. Create a folder named `data/CMAPSSData` in the project root.
3. Extract the contents of the downloaded zip folder into `data/CMAPSSData/`.
4. Ensure files like `train_FD001.txt`, `test_FD001.txt`, and `RUL_FD001.txt` are directly in `data/CMAPSSData/`.

*(Note: If you run our automated pipeline scripts, the dataset will be fetched and placed into the correct directories automatically).*

---

## Training the Models
Before launching the dashboard, you must preprocess the data and train the three AI models. Run the following scripts in order:

1. **Preprocess and Save Scaler**:
   ```bash
   python preprocess.py
   ```
2. **Train Model 1 — Anomaly Detector (Isolation Forest)**:
   ```bash
   python anomaly_model.py
   ```
3. **Train Model 2 — Remaining Useful Life (RUL) Regressor (Random Forest Regressor)**:
   ```bash
   python rul_model.py
   ```
4. **Train Model 3 — Fault Severity Classifier (Random Forest Classifier)**:
   ```bash
   python fault_classifier.py
   ```

Running these commands creates the `models/` directory and outputs:
*   `models/scaler.pkl`
*   `models/anomaly_model.pkl`
*   `models/rul_model.pkl`
*   `models/fault_classifier.pkl`

---

## Launching the Dashboard
Once the models are trained, run the Streamlit dashboard:
```bash
streamlit run app.py
```

### Dashboard Features:
*   **Neumorphism Dark Interface**: Custom soft-raised UI with glowing border indicators for alarms.
*   **Interactive Simulation controls**: Adjust replay delay, start/pause the telemetry stream, or reset simulation from the sidebar.
*   **Live Sparkline telemetry**: 4 key OBD-II readings (Temperature, Pressure, RPM, and Efficiency) render and update dynamically.
*   **Offline Engine Diagnostics**: The dashboard uses the pre-trained local `.pkl` files to score each cycle in real-time entirely offline.

---

## Offline Note
EdgeGuard is designed for offline deployment. The training scripts process data locally, and the dashboard runs fully without internet access once the dataset files are downloaded. No user data leaves the machine.
