"""
Post-Disruption Impact Analysis Modeling Module
Author: Lead AI/ML Developer
Project: AI-Based Supply Chain Disruption Prediction and Recovery System
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def train_impact_models():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    train_path = os.path.join(base_dir, "data", "processed", "train_2024.csv")
    test_path = os.path.join(base_dir, "data", "processed", "test_2025.csv")
    impact_csv_path = os.path.join(base_dir, "data", "processed", "impact_data.csv")
    
    models_dir = os.path.join(base_dir, "models")
    impact_models_path = os.path.join(models_dir, "impact_models.pkl")
    impact_metrics_path = os.path.join(models_dir, "impact_metrics.json")
    
    print("Loading train and test datasets for Impact Analysis...", flush=True)
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    
    # Extract Impact Dataset: Disrupted events (Disruption_Occurred == 1)
    impact_train = df_train[df_train['Disruption_Occurred'] == 1].copy()
    impact_test = df_test[df_test['Disruption_Occurred'] == 1].copy()
    
    print(f"Impact modeling subset: {len(impact_train):,} train events (2024), {len(impact_test):,} test events (2025)", flush=True)
    
    # Save combined impact data for downstream reference
    df_impact_all = pd.concat([impact_train, impact_test], axis=0).reset_index(drop=True)
    df_impact_all.to_csv(impact_csv_path, index=False)
    print(f"Saved impact dataset to: {impact_csv_path}", flush=True)
    
    # Features available at point of disruption diagnosis:
    # Operational, shipment, route, and disruption characteristics
    feature_cols = [
        'Distance_km', 'Weight_MT', 'Shipment_Quantity', 'Lead_Time_Days',
        'Weather_Risk_Score', 'Geopolitical_Risk_Score', 'Supplier_Reliability_Score',
        'Carrier_Reliability_Score', 'Capacity_Utilization', 'Inventory_Level', 'Safety_Stock'
    ]
    
    # Categorical features
    cat_cols = ['Transport_Mode', 'Product_Category', 'Disruption_Severity']
    
    X_train_num = impact_train[feature_cols].copy()
    X_test_num = impact_test[feature_cols].copy()
    
    X_train_cat = pd.get_dummies(impact_train[cat_cols], drop_first=False)
    cat_encoded_cols = X_train_cat.columns.tolist()
    
    X_test_cat = pd.get_dummies(impact_test[cat_cols], drop_first=False)
    X_test_cat = X_test_cat.reindex(columns=cat_encoded_cols, fill_value=0)
    
    X_train = pd.concat([X_train_num, X_train_cat], axis=1).astype(np.float32)
    X_test = pd.concat([X_test_num, X_test_cat], axis=1).astype(np.float32)
    
    # Targets for Impact Assessment
    targets = {
        'Delivery_Delay_Days': GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=42),
        'Inventory_Shortage_Units': RandomForestRegressor(n_estimators=80, max_depth=6, random_state=42, n_jobs=-1),
        'Financial_Loss_USD': GradientBoostingRegressor(n_estimators=120, max_depth=5, random_state=42),
        'Production_Impact_Pct': GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=42),
        'Overall_Impact_Score': GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=42)
    }
    
    trained_models = {}
    metrics_summary = {}
    
    print("\n--- Training Consequence & Impact Estimators ---", flush=True)
    for target_name, model in targets.items():
        print(f"Training estimator for: {target_name}...", flush=True)
        y_train = impact_train[target_name].values
        y_test = impact_test[target_name].values
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        mae = float(mean_absolute_error(y_test, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        r2 = float(r2_score(y_test, y_pred))
        
        metrics_summary[target_name] = {
            'mae': mae,
            'rmse': rmse,
            'r2_score': r2,
            'mean_actual': float(np.mean(y_test)),
            'std_actual': float(np.std(y_test))
        }
        trained_models[target_name] = model
        print(f"  {target_name} -> MAE: {mae:.2f}, RMSE: {rmse:.2f}, R²: {r2:.3f}", flush=True)
        
    # Save artifacts
    artifacts = {
        'models': trained_models,
        'feature_cols': feature_cols,
        'cat_cols': cat_cols,
        'cat_encoded_cols': cat_encoded_cols,
        'full_feature_names': X_train.columns.tolist()
    }
    joblib.dump(artifacts, impact_models_path)
    print(f"Saved impact models to: {impact_models_path}", flush=True)
    
    with open(impact_metrics_path, "w") as f:
        json.dump(metrics_summary, f, indent=2)
    print(f"Saved impact metrics to: {impact_metrics_path}", flush=True)
    
    return metrics_summary

if __name__ == "__main__":
    train_impact_models()
