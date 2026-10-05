# PROJECT FINAL REPORT
## AI-Based Supply Chain Disruption Prediction and Recovery System
**Tagline:** Predict. Simulate. Recover. Explain.  
**Role:** Lead AI/ML Developer  
**Status:** Completed & Validated

---

### A. Dataset Information
- **Original Source File:** `data/integrated_synthetic_supply_chain_dataset.xlsx`
- **Total Records:** 100,000 rows
- **Total Columns (Raw):** 63 columns
- **Workbook Sheets Identified:**
  1. `Integrated_Data`: 100,000 rows × 63 columns
  2. `Data_Dictionary`: 63 rows × 5 columns
  3. `Model_Mapping`: 7 rows × 4 columns
- **Date Range:** `2024-01-01` to `2025-12-30` (730 continuous calendar days)
- **Primary Classification Target:** `Disruption_Occurred` (Binary: 0 = No Disruption, 1 = Disruption)
- **Class Distribution:** 71,791 Class 0 (71.79%) vs. 28,209 Class 1 (28.21%)
- **Entity Cardinalities:**
  - Suppliers: 50 unique entities
  - Factories: 17 unique entities
  - Origin Ports: 5 unique entities
  - Destination Ports: 6 unique entities
  - Warehouses: 15 unique entities
  - Logistics Routes: 60 unique corridors
  - Logistics Carriers: 6 carriers
  - Products: 100 unique SKUs across 5 industry categories

---

### B. Cleaning Performed
- **Preservation of Raw Data:** The original Excel workbook was preserved unmodified.
- **Redundancy Analysis:**
  - `Production_Loss_Pct` was verified to be an exact duplicate of `Production_Impact_Pct` (Max absolute difference = 0.0, Pearson correlation = 1.0). Dropped in preprocessing.
  - `Delay_Days` was verified to be an exact duplicate of `Delivery_Delay_Days` (Max absolute difference = 0.0, Pearson correlation = 1.0). Dropped in preprocessing.
  - `Currency` was verified to be a constant string (`"USD"` across all 100,000 rows). Dropped in preprocessing.
- **Missing Value Imputation:**
  - `Disruption_Severity` contained exactly 71,791 missing values (`NaN`).
  - Analysis confirmed that 100% of these null values occurred when `Disruption_Occurred == 0`.
  - Imputed missing values with the category `"No Disruption"`.
- **Deduplication & Sorting:**
  - Zero exact duplicate rows were detected.
  - Confirmed 100,000 unique `Record_ID` and `Shipment_ID` values.
  - Standardized string casing and whitespace.
  - Sorted strictly chronologically by `Date` and `Record_ID`.
- **Output:** `data/processed/cleaned_supply_chain.csv` (100,000 rows × 60 clean columns).

---

### C. Train / Test Split
- **Methodology:** Strict chronological out-of-time partition. No random splitting or standard cross-validation leakage.
- **Training Set (Year 2024):**
  - File: `data/processed/train_2024.csv`
  - Records: 50,213 rows (2024-01-01 to 2024-12-31)
  - Class Distribution: 36,078 Class 0 (71.85%), 14,135 Class 1 (28.15%)
- **Test Set (Year 2025):**
  - File: `data/processed/test_2025.csv`
  - Records: 49,787 rows (2025-01-01 to 2025-12-30)
  - Class Distribution: 35,713 Class 0 (71.73%), 14,074 Class 1 (28.27%)
- **Integrity Validation:** Zero 2025 records in training set; zero 2024 records in test set; monotonic chronological order verified.

---

### D. Leakage Audit
A comprehensive audit script (`preprocessing/leakage_audit.py`) was executed. All 7 audit dimensions achieved a **PASS** rating:
1. **Temporal Leakage:** PASS (Strict separation; zero date overlap).
2. **Train/Test Contamination:** PASS (Zero overlapping `Record_ID` or `Shipment_ID` values).
3. **Target-Derived Variables:** PASS (`Disruption_Probability` identified as synthetic risk score and quarantined from pre-disruption feature space).
4. **Post-Disruption Leakage:** PASS (All 14 consequence fields strictly quarantined from XGBoost: `Disruption_Type`, `Disruption_Severity`, `Cargo_Condition`, `Order_Fulfillment_Status`, `Delivery_Time_Deviation`, `Delivery_Delay_Days`, `Inventory_Shortage_Units`, `Financial_Loss_USD`, `Production_Impact_Pct`, `Customer_Impact_Score`, `Overall_Impact_Score`, `Recovery_Strategy`, `Recovery_Cost_USD`, `Recovery_Time_Days`).
5. **Duplicate Feature Leakage:** PASS (All duplicate and constant columns removed).
6. **Encoder Leakage:** PASS (Frequency encodings and one-hot encoders fitted strictly on 2024 data).
7. **Scaler Leakage:** PASS (Normalization scalers fitted solely on training splits).
- **Report Location:** `preprocessing/leakage_audit_report.txt`

---

### E. XGBoost Pre-Disruption Features
Total Features in Final Feature Space: **94 Features**
- **Operational & Shipment Features:** `Distance_km`, `Weight_MT`, `Shipment_Quantity`, `Lead_Time_Days`, `Customs_Clearance_Time`, `Handling_Equipment_Availability`, `Capacity_Utilization`, `Fuel_Price_Index`, `Shipping_Cost_USD`, `Commodity_Price`.
- **Pre-Disruption Risk Ratings:** `Weather_Risk_Score`, `Geopolitical_Risk_Score`, `Supplier_Reliability_Score`, `Carrier_Reliability_Score`, `Supplier_Tier`.
- **Inventory & Operational State:** `Historical_Demand`, `Forecast_Horizon`, `Safety_Stock`, `Inventory_Level`.
- **Infrastructure Options (Pre-Event Capabilities):** `Backup_Supplier_Available`, `Rerouting_Available`, `Inventory_Reallocation_Possible`.
- **GNN Graph Features:** `Supplier_GNN_Risk`, `Factory_GNN_Risk`, `Origin_Port_Betweenness`, `Warehouse_PageRank`, `Network_Vulnerability_Index`.
- **Leakage-Safe Frequency Encodings (Fitted on Train Only):** `Product_ID_freq`, `Route_ID_freq`, `Supplier_ID_freq`, `Factory_ID_freq`, `Warehouse_ID_freq`, `Alternative_Supplier_ID_freq`.
- **One-Hot Encoded Categorical Attributes:** `Transport_Mode`, `Carrier`, `Product_Category`, `Supplier_Country`, `Route_Risk_Level`, `Port_Congestion_Level`, `Weather_Condition`, `Commodity`, `Commodity_Category`, `Origin_Port`, `Destination_Port`, `DayOfWeek`.

---

### F. XGBoost Model Metrics (Evaluated on 2025 Test Set)
- **Model File:** `models/xgboost_disruption_model.json`
- **Class Balancing:** `scale_pos_weight = 2.5524` (Computed from 2024 training distribution)
- **Metrics Summary:**
  - **Accuracy:** 0.6357 (63.57%)
  - **Precision:** 0.3880 (38.80%)
  - **Recall:** 0.4996 (49.96%)
  - **F1-Score:** 0.4368
  - **ROC-AUC:** 0.6312
  - **PR-AUC (Precision-Recall AUC):** 0.4039
  - **Confusion Matrix:**
    - True Negatives (TN): 24,619
    - False Positives (FP): 11,094
    - False Negatives (FN): 7,042
    - True Positives (TP): 7,032
- **Metrics File:** `models/xgboost_metrics.json`
- **Feature Importance Artifacts:** `models/plots/xgboost_feature_importance.png`, `models/xgboost_feature_importance.json`

---

### G. LSTM Demand Forecasting Features
- **Lookback Window:** 14 continuous days
- **Chronological Time Series:** 730 continuous daily time steps
- **Multivariate Features:**
  1. `Historical_Demand` (Observed daily demand units)
  2. `Inventory_Level` (Daily average on-hand stock)
  3. `Safety_Stock` (Daily average minimum buffer requirement)
  4. `Shipment_Quantity` (Daily average units in transit)
- **Target:** Next-day `Historical_Demand` ($t+1$)
- **Architecture:**
  - `Input(shape=(14, 4))`
  - `LSTM(64, return_sequences=True)`
  - `Dropout(0.2)`
  - `LSTM(32)`
  - `Dropout(0.2)`
  - `Dense(16, activation="relu")`
  - `Dense(1)`
- **Optimizer:** Adam (learning rate = 0.005)
- **Loss Function:** Mean Squared Error (MSE)

---

### H. LSTM Model Metrics (Evaluated on 2025 Test Set)
- **Model File:** `models/lstm_demand_model.keras`
- **Metrics Summary:**
  - **Mean Absolute Error (MAE):** 6.1802 units
  - **Mean Squared Error (MSE):** 70.8680
  - **Root Mean Squared Error (RMSE):** 8.4183 units
  - **Mean Absolute Percentage Error (MAPE):** 1.42%
  - **Coefficient of Determination ($R^2$):** 0.9633
- **History & Scaler Artifacts:**
  - `models/lstm_metrics.json`
  - `models/lstm_training_history.json`
  - `models/preprocessing_scaler.pkl`
- **Plots Generated:**
  - `models/plots/lstm_actual_vs_predicted.png`
  - `models/plots/lstm_training_loss.png`
  - `models/plots/lstm_residuals.png`

---

### I. GNN Structure & Results
- **Topology Extraction:** Directed 5-echelon supply-chain network.
- **Node Summary (93 Nodes):**
  - Tier 1: 50 Suppliers
  - Tier 2: 17 Factories
  - Tier 3: 5 Origin Ports
  - Tier 4: 6 Destination Ports
  - Tier 5: 15 Warehouses
- **Edge Summary (187 Edges):**
  - Supplier $\rightarrow$ Factory: 50 edges
  - Factory $\rightarrow$ Origin Port: 17 edges
  - Origin Port $\rightarrow$ Destination Port: 30 edges
  - Destination Port $\rightarrow$ Warehouse: 90 edges
- **Graph Centrality Computed:** In-Degree, Out-Degree, Betweenness Centrality, PageRank.
- **GNN Model:** 2-Layer Graph Convolutional Network (PyTorch) trained for 150 epochs.
- **Final Node Vulnerability MSE:** $0.000312$
- **Artifacts:**
  - `data/processed/nodes.csv`
  - `data/processed/edges.csv`
  - `models/gnn_network_model.pt`
  - `models/gnn_metrics.json`

---

### J. Impact Analysis Results
- **Dataset:** `data/processed/impact_data.csv` (Filtered on verified disruption events: 14,135 train, 14,074 test).
- **Target Estimators (Trained on 2024, Evaluated on 2025 Out-of-Time Test Events):**
  1. `Delivery_Delay_Days`: MAE = 1.01 days | RMSE = 1.17 days | $R^2 = 0.851$
  2. `Inventory_Shortage_Units`: MAE = 5.87 units | RMSE = 10.31 units | $R^2 = 0.899$
  3. `Financial_Loss_USD`: MAE = $188,030 | RMSE = $525,524 | $R^2 = 0.906$
  4. `Production_Impact_Pct`: MAE = 2.68% | RMSE = 3.35% | $R^2 = 0.787$
  5. `Overall_Impact_Score`: MAE = 2.64 / 100 | RMSE = 3.22 / 100 | $R^2 = 0.805$
- **Artifacts:** `models/impact_models.pkl`, `models/impact_metrics.json`

---

### K. Monte Carlo Scenarios & Simulation Results
- **Execution:** 2,000 iterations per scenario based on empirical historical distributions.
- **Scenarios Evaluated:**
  - **Scenario A: Continue Current Supplier (Wait & Expedite):**
    - Expected Delay: 1.2 days | Expected Cost: $129,428 | Shortage: 0.2 units | Risk Score: 12.4 / 100
  - **Scenario B: Switch to Alternative Supplier:**
    - Expected Delay: 20.9 days | Expected Cost: $4,033,964 | Shortage: 192.2 units | Risk Score: 61.3 / 100
  - **Scenario C: Shipment Rerouting:**
    - Expected Delay: 11.2 days | Expected Cost: $1,672,396 | Shortage: 54.4 units | Risk Score: 43.8 / 100
  - **Scenario D: Inventory Reallocation:**
    - Expected Delay: 7.0 days | Expected Cost: $923,244 | Shortage: 15.0 units | Risk Score: 34.6 / 100
- **Artifacts:**
  - `simulation/simulation_results/monte_carlo_summary.json`
  - `simulation/simulation_results/monte_carlo_scenarios_comparison.png`

---

### L. Recovery Optimization
- **Script:** `optimization/recovery_optimizer.py`
- **Output Dataset:** `data/processed/recovery_data.csv` (100,000 rows with recovery options and constraints).
- **Optimization Strategy:** Constrained Multi-Criteria Decision Analysis (MCDA) minimizing weighted operational loss:
  $$J = w_{\text{cost}} \cdot \text{NormCost} + w_{\text{time}} \cdot \text{NormTime} + w_{\text{shortage}} \cdot \text{NormShortage}$$
- **Constraint Handling:** Strictly flags and penalizes infeasible actions when `Backup_Supplier_Available == 0`, `Rerouting_Available == 0`, or safety stock is insufficient for inter-warehouse transfer.
- **Output Philosophy:** Designates the best feasible candidate as the *"recommended recovery scenario based on the configured objective,"* explicitly avoiding claims of absolute or universal optimality.

---

### M. SHAP Explainability Results
- **Module:** `explainability/shap_analysis.py`
- **Method:** TreeSHAP on XGBoost model with representative background test set.
- **Global Key Drivers:**
  - High Route Risk Level (`Route_Risk_Level_High`)
  - Low Port Machinery Availability (`Handling_Equipment_Availability`)
  - Elevated Weather Risk (`Weather_Risk_Score`)
  - High Geopolitical Tension (`Geopolitical_Risk_Score`)
  - Carrier & Supplier Reliability Ratings
- **Local Explanations:** Real-time generation of waterfall and attribution breakdowns returning top positive (risk-increasing) and top negative (risk-reducing) factors for any queried `Shipment_ID`.
- **Artifacts:**
  - `explainability/shap_outputs/shap_summary_plot.png`
  - `explainability/shap_outputs/shap_bar_plot.png`
  - `explainability/shap_outputs/global_shap_importance.json`

---

### N. QR / Barcode Functionality
- **Script:** `qr_barcode/shipment_identifier.py`
- **Capabilities:**
  1. Generates standard scannable QR images encoding `SHIPMENT:<Shipment_ID>`.
  2. Decodes QR image buffers via OpenCV `cv2.QRCodeDetector` with `pyzbar` fallback.
  3. Queries the operational database to populate shipment telemetry.
- **Validation:** Tested end-to-end; generated and decoded `qr_barcode/sample_shipment_qr.png` without failure.

---

### O. Dashboard Functionality
- **Main Script:** `dashboard/app.py`
- **Application Engine:** Streamlit
- **10 Core Pages Implemented:**
  1. Home & System Overview
  2. Shipment Search & QR Ingestion
  3. Demand Forecasting (LSTM)
  4. Disruption Risk Prediction (XGBoost)
  5. Impact Assessment
  6. Monte Carlo Scenario Simulator
  7. Recovery Optimization (MCDA)
  8. Explainability (SHAP)
  9. Supply Chain Graph / GNN Explorer
  10. Model Benchmarks & Leakage Audit

---

### P. Generated Files & Artifacts Directory Map
```
SupplyChainAI/
├── dashboard/
│   └── app.py
├── data/
│   ├── integrated_synthetic_supply_chain_dataset.xlsx  [ORIGINAL UNMODIFIED]
│   └── processed/
│       ├── cleaned_supply_chain.csv
│       ├── train_2024.csv
│       ├── test_2025.csv
│       ├── nodes.csv
│       ├── edges.csv
│       ├── impact_data.csv
│       ├── recovery_data.csv
│       ├── disruption_feature_columns.json
│       ├── encoding_mappings.json
│       ├── lstm_train_data.npz
│       └── lstm_test_data.npz
├── explainability/
│   ├── shap_analysis.py
│   └── shap_outputs/
│       ├── shap_summary_plot.png
│       ├── shap_bar_plot.png
│       └── global_shap_importance.json
├── models/
│   ├── train_xgboost.py
│   ├── train_lstm.py
│   ├── train_impact.py
│   ├── xgboost_disruption_model.json
│   ├── xgboost_metrics.json
│   ├── xgboost_feature_importance.json
│   ├── lstm_demand_model.keras
│   ├── lstm_metrics.json
│   ├── lstm_training_history.json
│   ├── preprocessing_scaler.pkl
│   ├── gnn_network_model.pt
│   ├── gnn_metrics.json
│   ├── impact_models.pkl
│   ├── impact_metrics.json
│   └── plots/
│       ├── xgboost_feature_importance.png
│       ├── lstm_actual_vs_predicted.png
│       ├── lstm_training_loss.png
│       └── lstm_residuals.png
├── optimization/
│   └── recovery_optimizer.py
├── preprocessing/
│   ├── dataset_analysis.py
│   ├── dataset_analysis_report.txt
│   ├── preprocess_data.py
│   ├── create_supply_chain_graph.py
│   ├── leakage_audit.py
│   └── leakage_audit_report.txt
├── qr_barcode/
│   ├── shipment_identifier.py
│   └── sample_shipment_qr.png
├── simulation/
│   ├── monte_carlo.py
│   └── simulation_results/
│       ├── monte_carlo_summary.json
│       └── monte_carlo_scenarios_comparison.png
├── README.md
├── requirements.txt
└── PROJECT_FINAL_REPORT.md
```

---

### Q. Errors / Warnings Encountered & Resolved
1. **Categorical Dtype in XGBoost (`ValueError: could not convert string to float: 'Monday'`):**
   - *Cause:* `DayOfWeek` was not partitioned into `low_card_cols` during one-hot dummy generation.
   - *Fix:* Enhanced automatic inspection in `models/train_xgboost.py` to identify all object/category columns and ensure complete one-hot dummy encoding.
2. **Missing `openpyxl`, `xgboost`, `shap`, `keras` in Python 3.13 Host:**
   - *Resolution:* Installed required wheels with pip; configured Keras 3 with the native PyTorch backend (`KERAS_BACKEND=torch`) to save `.keras` model format.

---

### R. Limitations
1. **Decision Support Focus:** The system provides analytical recommendations; it does not trigger autonomous warehouse replenishment orders without human review.
2. **Batch Telemetry:** The ingestion layer operates on batch transaction data and QR scans rather than live satellite AIS GPS coordinates.
3. **Synthetic Domain Data:** Built upon the provided 100,000-row synthetic benchmark.

---

### S. Future Improvements
- Connect live AIS vessel telemetry feeds and NOAA weather alert APIs.
- Deep reinforcement learning (Q-learning / PPO) for multi-objective supply reallocation.
- Containerized REST API endpoints (FastAPI) for ERP integration.

---

### T. How to Launch the Complete Project
To launch the interactive decision-support dashboard, run the following command in PowerShell / Command Prompt from the `SupplyChainAI/` directory:
```powershell
streamlit run dashboard/app.py
```
Open your web browser at:
```
http://localhost:8501
```
All models, graphs, simulators, and explainability artifacts are pre-computed, pre-trained, and ready for presentation.
