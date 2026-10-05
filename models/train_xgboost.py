"""
XGBoost Pre-Disruption Prediction Model Training Module
Author: Lead AI/ML Developer
Project: AI-Based Supply Chain Disruption Prediction and Recovery System
"""

import os
import json
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def train_xgboost_pipeline():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    train_path = os.path.join(base_dir, "data", "processed", "train_2024.csv")
    test_path = os.path.join(base_dir, "data", "processed", "test_2025.csv")
    nodes_path = os.path.join(base_dir, "data", "processed", "nodes.csv")
    
    models_dir = os.path.join(base_dir, "models")
    plots_dir = os.path.join(models_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    model_save_path = os.path.join(models_dir, "xgboost_disruption_model.json")
    metrics_save_path = os.path.join(models_dir, "xgboost_metrics.json")
    feat_cols_save_path = os.path.join(base_dir, "data", "processed", "disruption_feature_columns.json")
    encodings_save_path = os.path.join(base_dir, "data", "processed", "encoding_mappings.json")
    feat_imp_save_path = os.path.join(models_dir, "xgboost_feature_importance.json")
    feat_imp_plot_path = os.path.join(plots_dir, "xgboost_feature_importance.png")
    
    print("Loading 2024 train and 2025 test datasets...", flush=True)
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    
    # 1. Target column
    target_col = "Disruption_Occurred"
    y_train = df_train[target_col].values
    y_test = df_test[target_col].values
    
    # 2. Strict Quarantining of Post-Disruption, Proxy & Identifier Columns
    quarantined_cols = [
        # Identifiers
        'Record_ID', 'Shipment_ID', 'Date',
        # Target proxy / leakage
        'Disruption_Probability',
        # Target
        'Disruption_Occurred',
        # Post-disruption consequences
        'Disruption_Type', 'Disruption_Severity', 'Cargo_Condition', 'Order_Fulfillment_Status',
        'Delivery_Time_Deviation', 'Delivery_Delay_Days', 'Delay_Days', 'Inventory_Shortage_Units',
        'Financial_Loss_USD', 'Production_Impact_Pct', 'Production_Loss_Pct', 'Customer_Impact_Score',
        'Overall_Impact_Score',
        # Recovery choices & outcomes
        'Recovery_Strategy', 'Recovery_Cost_USD', 'Recovery_Time_Days',
        # Constant / redundant
        'Currency'
    ]
    
    # Check what features remain
    candidate_features = [c for c in df_train.columns if c not in quarantined_cols]
    print(f"Candidate pre-disruption features count: {len(candidate_features)}", flush=True)
    
    # 3. Load GNN Network Features
    if os.path.exists(nodes_path):
        print("Enriching features with GNN node vulnerability and topology metrics...", flush=True)
        df_nodes = pd.read_csv(nodes_path)
        node_vulnerability = dict(zip(df_nodes['node_id'], df_nodes['gnn_vulnerability_score']))
        node_betweenness = dict(zip(df_nodes['node_id'], df_nodes['betweenness_centrality']))
        node_pagerank = dict(zip(df_nodes['node_id'], df_nodes['pagerank']))
        
        for df_curr in [df_train, df_test]:
            df_curr['Supplier_GNN_Risk'] = df_curr['Supplier_ID'].map(node_vulnerability).fillna(0.0)
            df_curr['Factory_GNN_Risk'] = df_curr['Factory_ID'].map(node_vulnerability).fillna(0.0)
            df_curr['Origin_Port_Betweenness'] = df_curr['Origin_Port'].map(node_betweenness).fillna(0.0)
            df_curr['Warehouse_PageRank'] = df_curr['Warehouse_ID'].map(node_pagerank).fillna(0.0)
            df_curr['Network_Vulnerability_Index'] = (df_curr['Supplier_GNN_Risk'] + df_curr['Factory_GNN_Risk']) / 2.0
            
        candidate_features.extend([
            'Supplier_GNN_Risk', 'Factory_GNN_Risk', 'Origin_Port_Betweenness',
            'Warehouse_PageRank', 'Network_Vulnerability_Index'
        ])
    
    # 4. Partition Categorical Columns into High-Cardinality and Low-Cardinality
    all_cat_cols = df_train[candidate_features].select_dtypes(include=['object', 'category']).columns.tolist()
    
    high_card_cols = ['Product_ID', 'Route_ID', 'Supplier_ID', 'Factory_ID', 'Warehouse_ID', 'Alternative_Supplier_ID']
    high_card_cols = [c for c in high_card_cols if c in candidate_features]
    
    # Any other categorical column is low-cardinality (including DayOfWeek, Transport_Mode, etc.)
    low_card_cols = [c for c in all_cat_cols if c not in high_card_cols]
    
    print(f"Categorical features: {len(all_cat_cols)} total ({len(high_card_cols)} high-card, {len(low_card_cols)} low-card)", flush=True)
    print("Low card categorical columns:", low_card_cols, flush=True)
    
    encoding_mappings = {'frequency_encodings': {}, 'one_hot_columns': []}
    
    # 5. Leakage-Safe Frequency Encoding for High-Cardinality Variables (Fitted on 2024 TRAIN only)
    for col in high_card_cols:
        freq_map = (df_train[col].value_counts() / len(df_train)).to_dict()
        encoding_mappings['frequency_encodings'][col] = freq_map
        
        # Apply mapping (unseen in test receives 0.0)
        df_train[col + '_freq'] = df_train[col].map(freq_map).fillna(0.0)
        df_test[col + '_freq'] = df_test[col].map(freq_map).fillna(0.0)
        
    # Numerical and binary features
    base_numeric_cols = [
        c for c in candidate_features 
        if c not in all_cat_cols
    ]
    freq_encoded_cols = [c + '_freq' for c in high_card_cols]
    
    X_train_num = df_train[base_numeric_cols + freq_encoded_cols].copy()
    X_test_num = df_test[base_numeric_cols + freq_encoded_cols].copy()
    
    # Fit One-Hot Encoding on 2024 Train only
    X_train_cat = pd.get_dummies(df_train[low_card_cols], drop_first=False)
    one_hot_cols = X_train_cat.columns.tolist()
    encoding_mappings['one_hot_columns'] = one_hot_cols
    
    # Align test set to training one-hot columns exactly
    X_test_cat = pd.get_dummies(df_test[low_card_cols], drop_first=False)
    X_test_cat = X_test_cat.reindex(columns=one_hot_cols, fill_value=0)
    
    # Combine final feature frames
    X_train = pd.concat([X_train_num, X_train_cat], axis=1)
    X_test = pd.concat([X_test_num, X_test_cat], axis=1)
    
    # Convert all columns to float32
    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)
    
    final_feature_names = X_train.columns.tolist()
    print(f"Final feature space dimension: {len(final_feature_names)} features.", flush=True)
    
    # Save feature names and encodings
    with open(feat_cols_save_path, "w") as f:
        json.dump(final_feature_names, f, indent=2)
    with open(encodings_save_path, "w") as f:
        json.dump(encoding_mappings, f, indent=2)
        
    # 6. Compute Class Imbalance Weight (scale_pos_weight) on 2024 TRAIN ONLY
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale_pos_weight = float(neg_count / pos_count)
    print(f"Training Class Distribution: 0 = {neg_count:,}, 1 = {pos_count:,} | scale_pos_weight = {scale_pos_weight:.4f}", flush=True)
    
    # 7. Train XGBoost Classifier
    print("Training XGBoost Classifier...", flush=True)
    clf = xgb.XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric='logloss',
        tree_method='hist'
    )
    clf.fit(X_train, y_train)
    
    # 8. Evaluation on 2025 Test Set
    print("\n--- Evaluating on 2025 Chronological Test Data ---", flush=True)
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]
    
    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred))
    rec = float(recall_score(y_test, y_pred))
    f1 = float(f1_score(y_test, y_pred))
    roc_auc = float(roc_auc_score(y_test, y_prob))
    pr_auc = float(average_precision_score(y_test, y_prob))
    cm = confusion_matrix(y_test, y_pred).tolist()
    
    metrics = {
        'model_name': 'XGBoost Disruption Classifier',
        'train_year': 2024,
        'test_year': 2025,
        'train_records': len(X_train),
        'test_records': len(X_test),
        'scale_pos_weight': scale_pos_weight,
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'confusion_matrix': cm
    }
    
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")
    print(f"PR-AUC:    {pr_auc:.4f}")
    print("Confusion Matrix:", cm)
    
    with open(metrics_save_path, "w") as f:
        json.dump(metrics, f, indent=2)
        
    # Save Model in JSON format
    clf.save_model(model_save_path)
    print(f"Saved model to: {model_save_path}", flush=True)
    
    # 9. Feature Importance Extraction and Plot
    importance_scores = clf.feature_importances_
    feat_imp = sorted(zip(final_feature_names, importance_scores), key=lambda x: x[1], reverse=True)
    feat_imp_dict = {f: float(s) for f, s in feat_imp}
    
    with open(feat_imp_save_path, "w") as f:
        json.dump(feat_imp_dict, f, indent=2)
        
    # Plot top 20 features
    top_20 = feat_imp[:20]
    top_names = [x[0] for x in reversed(top_20)]
    top_scores = [x[1] for x in reversed(top_20)]
    
    plt.figure(figsize=(10, 8))
    plt.barh(range(len(top_names)), top_scores, color='#1E88E5', align='center')
    plt.yticks(range(len(top_names)), top_names, fontsize=9)
    plt.xlabel('Relative Feature Importance (Gain / Weight)', fontsize=11)
    plt.title('Top 20 Pre-Disruption Predictive Features (XGBoost)', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(feat_imp_plot_path, dpi=200)
    plt.close()
    print(f"Saved feature importance plot to: {feat_imp_plot_path}", flush=True)
    
    return metrics

if __name__ == "__main__":
    train_xgboost_pipeline()
