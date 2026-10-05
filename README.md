# AI-Based Supply Chain Disruption Prediction and Recovery System

> **Tagline:** *Predict. Simulate. Recover. Explain.*  
> **System Type:** Enterprise AI-Driven Decision-Support System (DSS)  
> **Evaluation Framework:** Chronological Out-of-Time Train/Test Validation (2024 Train / 2025 Test)

---

## 1. Project Title
**AI-Based Supply Chain Disruption Prediction and Recovery System**

## 2. Problem Statement
Global supply chain networks face severe operational vulnerabilities stemming from extreme weather, geopolitical volatility, labor strikes, port congestion, and mechanical failures. Traditional Enterprise Resource Planning (ERP) systems operate reactively—logging delays only after bottlenecks materialize. Such delays propagate rapidly across multi-echelon tiers (suppliers, factories, maritime transit corridors, destination ports, and regional distribution centers), causing millions of dollars in financial loss and severe inventory shortfalls.

This project delivers a proactive, end-to-end AI decision-support system that identifies shipments, forecasts future macro demand, models network structural risk, predicts pre-disruption likelihood without data leakage, quantifies post-disruption operational consequences, simulates recovery scenarios under uncertainty, recommends constraint-aware mitigation actions, and provides game-theoretic explainability.

## 3. Objectives
1. **Identification**: Ingest and track shipments via automated QR/barcode encoding and decoding.
2. **Data Cleaning & Validation**: Clean and audit 100,000 supply-chain records, ensuring strict zero-leakage standards.
3. **Demand Forecasting**: Predict chronological daily consumption trends using a deep multi-layer LSTM network with a 14-day lookback window.
4. **Graph Modeling**: Model structural relationships across 5 supply-chain tiers using a Graph Neural Network (GNN) and network centrality measures.
5. **Disruption Prediction**: Classify pre-disruption risk using an XGBoost classifier with frequency encodings and class-imbalance weighting.
6. **Impact Assessment**: Estimate downstream consequences (delay days, inventory shortfalls, and financial losses).
7. **Monte Carlo Simulation**: Simulate uncertainty across 4 strategic recovery scenarios over 2,000 iterations.
8. **Recovery Optimization**: Recommend the best recovery action using multi-criteria decision analysis (MCDA).
9. **Explainable AI**: Unpack disruption risk drivers using TreeSHAP global and local feature attributions.
10. **Interactive Decision Dashboard**: Deliver a unified Streamlit dashboard for enterprise operations and demonstrations.

---

## 4. System Architecture
```
                         PHYSICAL SHIPMENT IDENTIFICATION
                             (QR / Barcode Scanning)
                                       ↓
                             DATA PREPROCESSING
             (Deduplication, Imputation, Leakage Quarantining)
                                       ↓
                    STRICT CHRONOLOGICAL TRAIN/TEST SPLIT
                        (2024 Train / 2025 Test)
                                       ↓
            ┌──────────────────────────┴──────────────────────────┐
            ↓                                                     ↓
   DEEP LSTM FORECASTER                                SUPPLY CHAIN GRAPH / GNN
  (14-Day Rolling Window)                            (5-Tier Risk Propagation)
            ↓                                                     ↓
            └──────────────────────────┬──────────────────────────┘
                                       ↓
                            XGBOOST PRE-DISRUPTION
                               RISK PREDICTION
                                       ↓
                            POST-DISRUPTION IMPACT
                                  ASSESSMENT
                                       ↓
                             MONTE CARLO SCENARIO
                             UNCERTAINTY SIMULATION
                                       ↓
                           CONSTRAINED DECISION
                                OPTIMIZATION
                                       ↓
                              TREESHAP GAME-THEORETIC
                                  EXPLAINABILITY
                                       ↓
                           STREAMLIT DECISION SUPPORT
                                   DASHBOARD
```

---

## 5. Dataset Description
- **Primary Source**: `data/integrated_synthetic_supply_chain_dataset.xlsx` (Only dataset used; never modified or replaced).
- **Dimensions**: Exactly 100,000 records × 63 columns.
- **Coverage**: Continuous daily transactions from `2024-01-01` to `2025-12-30` (730 continuous days).
- **Workbook Sheets**:
  1. `Integrated_Data`: Core transaction and telemetry records (100,000 rows).
  2. `Data_Dictionary`: Semantic definitions, data types, and modules for all 63 fields.
  3. `Model_Mapping`: Formal specification of input columns, targets, and objectives per model.
- **Entity Counts**: 50 Suppliers, 17 Factories, 5 Origin Ports, 6 Destination Ports, 15 Warehouses, 60 Routes, 6 Carriers, 100 Products across 5 Categories.
- **Class Balance**: 71,791 Non-Disrupted (71.8%) vs. 28,209 Disrupted (28.2%).

---

## 6. Technologies Used
- **Programming Language**: Python 3.13.1 (64-bit)
- **Data Engineering**: `pandas`, `numpy`, `openpyxl`, `joblib`
- **Machine Learning**: `scikit-learn`, `xgboost` (v3.4.1)
- **Deep Learning**: `keras` (v3.15.1 with `torch` PyTorch backend)
- **Graph & Network Analysis**: `networkx` (v3.5), `torch` (v2.8.0)
- **Explainability**: `shap` (v0.52.0 - TreeExplainer)
- **Computer Vision & Ingestion**: `opencv-python`, `qrcode`, `pyzbar`, `Pillow`
- **Visualization & UI**: `streamlit` (v1.49.1), `matplotlib`

---

## 7. Machine Learning & Deep Learning Models
| Component | Algorithm / Framework | Key Hyperparameters / Design | Primary Objective |
| :--- | :--- | :--- | :--- |
| **Demand Forecaster** | 2-Layer Deep LSTM (Keras / PyTorch) | Lookback=14d, Units=[64, 32], Dropout=0.2, Dense(16, relu), Adam lr=0.005 | Daily aggregate demand prediction ($R^2 = 0.963$) |
| **Network Topology** | 2-Layer Graph Convolutional Network (PyTorch) | GCN Conv1(6→16), Conv2(16→8), Risk Head(8→1), lr=0.01 | Node vulnerability & structural risk propagation |
| **Disruption Classifier** | XGBoost Hist Classifier | Trees=300, Depth=6, lr=0.05, `scale_pos_weight`=2.5524, colsample=0.8 | Pre-disruption risk classification (PR-AUC=0.404) |
| **Impact Assessment** | Gradient Boosting & Random Forest Regressors | Depth=[4, 6], Trees=[80, 120], Loss='squared_error' | Post-event delay, stockout, and financial loss estimation |
| **Scenario Simulator** | Monte Carlo Engine (NumPy) | 2,000 iterations, parametric empirical distributions | Quantify financial VaR and delay distributions under uncertainty |
| **Decision Optimizer** | Multi-Criteria Decision Analysis (MCDA) | Configurable weights ($w_{\text{cost}}, w_{\text{time}}, w_{\text{shortage}}$) | Constraint-aware recovery recommendation |
| **Explainability** | TreeSHAP Explainer | Background sample=400, game-theoretic Shapley values | Localized waterfall and global beeswarm feature attributions |

---

## 8. Preprocessing & Leakage Quarantining
- **Deduplication**: Pruned redundant columns `Delay_Days` (identical to `Delivery_Delay_Days`), `Production_Loss_Pct` (identical to `Production_Impact_Pct`), and constant `Currency` (`"USD"`).
- **Imputation**: Transformed 71,791 structural missing values in `Disruption_Severity` to `"No Disruption"`.
- **Temporal Integrity**: Sorted chronologically by `Date` and `Record_ID`.
- **Strict Leakage Quarantining**: To prevent target leakage, 17 post-disruption variables were strictly quarantined from the pre-disruption XGBoost model:
  `Disruption_Probability`, `Disruption_Type`, `Disruption_Severity`, `Cargo_Condition`, `Order_Fulfillment_Status`, `Delivery_Time_Deviation`, `Delivery_Delay_Days`, `Inventory_Shortage_Units`, `Financial_Loss_USD`, `Production_Impact_Pct`, `Customer_Impact_Score`, `Overall_Impact_Score`, `Recovery_Strategy`, `Recovery_Cost_USD`, `Recovery_Time_Days`.
- **Encoding Safety**: High-cardinality entities (`Product_ID`, `Route_ID`, `Supplier_ID`, `Factory_ID`, `Warehouse_ID`) were frequency-encoded strictly using 2024 training frequencies. Unseen test entities receive 0.0.

---

## 9. LSTM Methodology
- Daily network demand was chronologically aggregated over 730 continuous days.
- Multivariate sequence tensors of shape `(Batch, 14, 4)` were constructed using: `Historical_Demand`, `Inventory_Level`, `Safety_Stock`, `Shipment_Quantity`.
- Normalization: `MinMaxScaler` fitted exclusively on 2024 records (`models/preprocessing_scaler.pkl`).
- Training: 60 epochs with early validation monitoring.
- Evaluation on 2025 out-of-time test set:
  - **MAE**: 6.18 units
  - **RMSE**: 8.42 units
  - **MAPE**: 1.42%
  - **$R^2$ Score**: 0.9633

---

## 10. XGBoost Methodology
- Trained on 50,213 records (Year 2024) to predict binary target `Disruption_Occurred`.
- High class-imbalance ratio ($2.55 : 1$) compensated by setting `scale_pos_weight = 2.5524` from training distribution.
- 94 clean pre-disruption features (operational, environmental, shipment, and GNN graph centrality metrics).
- Evaluated on 49,787 out-of-time records (Year 2025):
  - **Accuracy**: 63.57%
  - **Precision**: 38.80%
  - **Recall**: 49.96%
  - **F1-Score**: 43.68%
  - **ROC-AUC**: 0.6312
  - **PR-AUC**: 0.4039

---

## 11. GNN Methodology
- **Nodes (93)**: 50 Suppliers $\rightarrow$ 17 Factories $\rightarrow$ 5 Origin Ports $\rightarrow$ 6 Destination Ports $\rightarrow$ 15 Warehouses.
- **Edges (187)**: Directed logistics flow corridors across the 4 inter-echelon hops.
- Graph Convolutional Network trained for 150 epochs using normalized adjacency $\tilde{D}^{-1/2}\tilde{A}\tilde{D}^{-1/2}$ and 6 node features (Echelon, In-Degree, Out-Degree, Betweenness, PageRank, Operational Reliability).
- Final node risk MSE: $0.000312$.
- Extracted node vulnerability scores and passed them into the pre-disruption XGBoost feature space.

---

## 12. Impact Analysis
- Analyzes post-disruption operational consequences using empirical regressors trained on historical disrupted records:
  - `Delivery_Delay_Days` ($R^2 = 0.851$, MAE = 1.01 days)
  - `Inventory_Shortage_Units` ($R^2 = 0.899$, MAE = 5.87 units)
  - `Financial_Loss_USD` ($R^2 = 0.906$, MAE = $188,000)
  - `Production_Impact_Pct` ($R^2 = 0.787$, MAE = 2.68%)
  - `Overall_Impact_Score` ($R^2 = 0.805$, MAE = 2.64 / 100)

---

## 13. Monte Carlo Simulation
- Evaluates 4 candidate recovery strategies over 2,000 stochastic runs:
  - **Scenario A**: Continue with Current Supplier (Wait & Expedite)
  - **Scenario B**: Switch to Alternative Supplier
  - **Scenario C**: Shipment Rerouting
  - **Scenario D**: Inventory Reallocation
- Parameterized with empirical means and variances derived directly from historical recovery data.
- Dynamically scales distributions based on target shipment quantity, route distance, and scheduled lead time.
- Outputs expected loss, delay days, and 95% Value-at-Risk (VaR).

---

## 14. Recovery Optimization (MCDA)
- Constrained Multi-Criteria Decision Analysis evaluating feasible recovery actions under operational constraints:
  - `Backup_Supplier_Available`
  - `Rerouting_Available`
  - `Inventory_Reallocation_Possible`
- Minimizes a weighted loss objective:
  $$J = w_{\text{cost}} \cdot \frac{\text{Cost}}{\text{Cost}_{\text{max}}} + w_{\text{time}} \cdot \frac{\text{Time}}{\text{Time}_{\text{max}}} + w_{\text{shortage}} \cdot \frac{\text{Shortage}}{\text{Shortage}_{\text{max}}}$$
- Penalizes infeasible actions and outputs a transparent recommendation with full trade-off explanations.

---

## 15. SHAP Explainability
- Uses TreeSHAP to attribute game-theoretic contributions to disruption predictions.
- Global level: Beeswarm and bar plots identifying top macro risk drivers (Route Risk, Weather Risk, Handling Equipment Uptime, Supplier Reliability).
- Local level: Decomposes predictions for individual shipments into factors that increase risk vs. factors that reduce risk.

---

## 16. QR / Barcode Module
- `qr_barcode/shipment_identifier.py` provides non-invasive physical identification.
- Encodes `Shipment_ID` into scannable QR images (`sample_shipment_qr.png`).
- Decodes image buffers using OpenCV `cv2.QRCodeDetector` and `pyzbar`.
- Queries the backend database to populate operational variables instantly.

---

## 17. Streamlit Dashboard
The web application is located at `dashboard/app.py` and features 10 comprehensive pages:
1. **Home & Overview**: Architecture diagram, dataset dimensions, and methodology.
2. **Shipment Search & QR**: Live search, preset selection, QR code generation, and image scanning.
3. **Demand Forecasting (LSTM)**: Interactive charts comparing actual vs. predicted 2025 demand, loss curves, and residuals.
4. **Disruption Risk (XGBoost)**: Risk probability gauge, Low/Medium/High classification, and operational factor cards.
5. **Impact Assessment**: Estimations for delivery delay, inventory shortfall, and financial damage.
6. **Monte Carlo Simulator**: 2,000-run simulation histograms, 95% VaR, and strategy comparisons.
7. **Recovery Optimization**: Interactive weight sliders ($w_{\text{cost}}, w_{\text{time}}, w_{\text{shortage}}$) and constraint feasibility tables.
8. **Explainability (SHAP)**: Individualized factor cards (increasing vs. decreasing risk) and global beeswarm plots.
9. **Supply Chain Graph / GNN**: 5-echelon node counts, connection matrices, and top vulnerable entities.
10. **Model Benchmarks & Audit**: Scorecards for XGBoost and LSTM alongside the official Leakage Audit Report.

---

## 18. Results Summary
- **Dataset**: 100,000 rows × 60 clean columns (730 continuous calendar days).
- **Leakage Audit**: 7/7 Categories **PASS** (Zero future leakage, zero target proxy leakage).
- **LSTM Demand Forecaster**: $R^2 = 0.963$, MAE = 6.18 units, MAPE = 1.42%.
- **XGBoost Classifier**: ROC-AUC = 0.6312, PR-AUC = 0.4039, Recall = 49.96%.
- **GNN Topology**: 93 nodes, 187 directed edges, MSE = $0.000312$.
- **Impact Regressors**: $R^2 \ge 0.787$ across all 5 operational consequence metrics.
- **End-to-End Test**: 100% test passage across diverse 2024 and 2025 shipments.

---

## 19. Limitations
1. **Decision Support Nature**: The system provides algorithmic recommendations based on configured objectives and historical patterns; human managerial review is required before taking physical action.
2. **No Real-Time Telemetry Tracking**: Physical GPS/AIS vessel positions are not streamed live; data is ingested from structured batch transactions and QR scans.
3. **Synthetic Dataset Baseline**: The dataset is a high-fidelity synthetic benchmark reflecting real-world statistical correlations.

---

## 20. Future Enhancements
- Integration with real-time AIS maritime tracking and satellite weather APIs.
- Extension to dynamic reinforcement learning (RL) for automated multi-echelon replenishment.
- Deployment in containerized microservices (Docker + Kubernetes).

---

## 21. How to Run the Project

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: (Optional) Re-run Pipeline Stages
All models and artifacts are pre-trained and saved. To reproduce from scratch:
```bash
# 1. Dataset Analysis
python preprocessing/dataset_analysis.py

# 2. Data Cleaning & Chronological Split
python preprocessing/preprocess_data.py

# 3. Supply Chain Graph & GNN Training
python preprocessing/create_supply_chain_graph.py

# 4. Leakage Audit
python preprocessing/leakage_audit.py

# 5. XGBoost Model Training
python models/train_xgboost.py

# 6. LSTM Demand Forecasting Model Training
python models/train_lstm.py

# 7. Impact Consequence Analysis Training
python models/train_impact.py

# 8. Monte Carlo Simulation
python simulation/monte_carlo.py

# 9. Recovery Optimization
python optimization/recovery_optimizer.py

# 10. SHAP Explainability Generation
python explainability/shap_analysis.py

# 11. QR Code Module Verification
python qr_barcode/shipment_identifier.py
```

### Step 3: Launch the Streamlit Dashboard
```bash
streamlit run dashboard/app.py
```
The dashboard will open automatically in your browser at `http://localhost:8501`.
