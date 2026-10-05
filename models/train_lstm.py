"""
LSTM Demand Forecasting Model Training Module
Author: Lead AI/ML Developer
Project: AI-Based Supply Chain Disruption Prediction and Recovery System
"""

import os
os.environ["KERAS_BACKEND"] = "torch"
import json
import joblib
import pandas as pd
import numpy as np
import keras
from keras import layers
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Reproducibility seeds
np.random.seed(42)
keras.utils.set_random_seed(42)

def train_lstm_pipeline():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cleaned_path = os.path.join(base_dir, "data", "processed", "cleaned_supply_chain.csv")
    models_dir = os.path.join(base_dir, "models")
    plots_dir = os.path.join(models_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    model_save_path = os.path.join(models_dir, "lstm_demand_model.keras")
    metrics_save_path = os.path.join(models_dir, "lstm_metrics.json")
    history_save_path = os.path.join(models_dir, "lstm_training_history.json")
    scaler_save_path = os.path.join(models_dir, "preprocessing_scaler.pkl")
    
    print(f"Loading cleaned data from {cleaned_path}...", flush=True)
    df = pd.read_csv(cleaned_path)
    df['Date'] = pd.to_datetime(df['Date'])
    
    # 1. Aggregate Daily Demand and Inventory Telemetry Chronologically
    daily_df = df.groupby('Date').agg({
        'Historical_Demand': 'mean',
        'Inventory_Level': 'mean',
        'Safety_Stock': 'mean',
        'Shipment_Quantity': 'mean'
    }).sort_index().reset_index()
    
    print(f"Total continuous daily time steps: {len(daily_df)} days ({daily_df['Date'].min().strftime('%Y-%m-%d')} to {daily_df['Date'].max().strftime('%Y-%m-%d')})", flush=True)
    
    # 2. Chronological Split (Train: 2024, Test: 2025)
    train_mask = daily_df['Date'].dt.year == 2024
    test_mask = daily_df['Date'].dt.year == 2025
    
    train_daily = daily_df[train_mask].copy().reset_index(drop=True)
    test_daily = daily_df[test_mask].copy().reset_index(drop=True)
    
    feature_cols = ['Historical_Demand', 'Inventory_Level', 'Safety_Stock', 'Shipment_Quantity']
    target_idx = feature_cols.index('Historical_Demand')
    
    # 3. Fit Scaler Exclusively on 2024 Training Set
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_scaled = scaler.fit_transform(train_daily[feature_cols].values)
    
    # Transform test set using train-fitted parameters
    # To predict the first days of 2025 with 14-day lookback, prepend last 14 days of 2024
    lookback = 14
    test_full_values = np.vstack([train_daily[feature_cols].values[-lookback:], test_daily[feature_cols].values])
    test_scaled = scaler.transform(test_full_values)
    
    joblib.dump(scaler, scaler_save_path)
    print(f"Saved MinMaxScaler to: {scaler_save_path}", flush=True)
    
    # 4. Create Sliding Window Sequences (No random shuffling)
    def create_sequences(data, lookback_window, target_col_idx):
        X, y = [], []
        for i in range(len(data) - lookback_window):
            X.append(data[i:i + lookback_window, :])
            y.append(data[i + lookback_window, target_col_idx])
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)
        
    X_train, y_train = create_sequences(train_scaled, lookback, target_idx)
    X_test, y_test = create_sequences(test_scaled, lookback, target_idx)
    
    print(f"Constructed sequence tensors: X_train shape: {X_train.shape}, y_train shape: {y_train.shape}", flush=True)
    print(f"Constructed sequence tensors: X_test shape: {X_test.shape}, y_test shape: {y_test.shape}", flush=True)
    
    # Save processed numpy arrays
    np.savez_compressed(os.path.join(base_dir, "data", "processed", "lstm_train_data.npz"), X=X_train, y=y_train)
    np.savez_compressed(os.path.join(base_dir, "data", "processed", "lstm_test_data.npz"), X=X_test, y=y_test)
    
    # 5. Build Model Architecture as Specified
    # LSTM(64, return_sequences=True) -> Dropout(0.2) -> LSTM(32) -> Dropout(0.2) -> Dense(16, relu) -> Dense(1)
    model = keras.Sequential([
        layers.Input(shape=(lookback, len(feature_cols))),
        layers.LSTM(64, return_sequences=True),
        layers.Dropout(0.2),
        layers.LSTM(32),
        layers.Dropout(0.2),
        layers.Dense(16, activation="relu"),
        layers.Dense(1)
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.005),
        loss="mse",
        metrics=["mae"]
    )
    model.summary()
    
    # 6. Train Chronologically (Validation split chronologically from end of train)
    val_size = int(len(X_train) * 0.15)
    X_tr, y_tr = X_train[:-val_size], y_train[:-val_size]
    X_val, y_val = X_train[-val_size:], y_train[-val_size:]
    
    print("Training LSTM Demand Model...", flush=True)
    history = model.fit(
        X_tr, y_tr,
        validation_data=(X_val, y_val),
        epochs=60,
        batch_size=16,
        verbose=1
    )
    
    # Save Model
    model.save(model_save_path)
    print(f"Model saved cleanly to: {model_save_path}", flush=True)
    
    # Save training history
    hist_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    with open(history_save_path, "w") as f:
        json.dump(hist_dict, f, indent=2)
        
    # 7. Evaluate on 2025 Chronological Test Data
    print("\n--- Evaluating on 2025 Test Demand ---", flush=True)
    y_pred_scaled = model.predict(X_test)
    
    # Inverse transform to original demand units
    dummy_test = np.zeros((len(y_test), len(feature_cols)))
    dummy_pred = np.zeros((len(y_pred_scaled), len(feature_cols)))
    
    dummy_test[:, target_idx] = y_test
    dummy_pred[:, target_idx] = y_pred_scaled.flatten()
    
    y_test_actual = scaler.inverse_transform(dummy_test)[:, target_idx]
    y_pred_actual = scaler.inverse_transform(dummy_pred)[:, target_idx]
    
    mae = float(mean_absolute_error(y_test_actual, y_pred_actual))
    mse = float(mean_squared_error(y_test_actual, y_pred_actual))
    rmse = float(np.sqrt(mse))
    mape = float(np.mean(np.abs((y_test_actual - y_pred_actual) / np.maximum(y_test_actual, 1e-5))) * 100)
    r2 = float(r2_score(y_test_actual, y_pred_actual))
    
    print(f"LSTM 2025 Test Performance:")
    print(f"  MAE:  {mae:.4f} units")
    print(f"  MSE:  {mse:.4f}")
    print(f"  RMSE: {rmse:.4f} units")
    print(f"  MAPE: {mape:.2f}%")
    print(f"  R²:   {r2:.4f}")
    
    metrics = {
        'model_name': 'LSTM Demand Forecaster',
        'lookback_days': lookback,
        'features': feature_cols,
        'train_year': 2024,
        'test_year': 2025,
        'train_sequences': len(X_train),
        'test_sequences': len(X_test),
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'mape': mape,
        'r2_score': r2
    }
    with open(metrics_save_path, "w") as f:
        json.dump(metrics, f, indent=2)
        
    # 8. Generate Required Evaluation Plots
    # Plot 1: Actual vs Predicted Demand
    plt.figure(figsize=(14, 5))
    test_dates = test_daily['Date'].values
    plt.plot(test_dates, y_test_actual, label='Actual 2025 Demand', color='#1565C0', alpha=0.85, linewidth=1.5)
    plt.plot(test_dates, y_pred_actual, label='LSTM Predicted Demand', color='#D32F2F', linestyle='--', linewidth=1.5)
    plt.title(f'LSTM Demand Forecast Evaluation (2025 Out-of-Time Test Set) | R² = {r2:.3f}', fontsize=12, fontweight='bold')
    plt.xlabel('Date')
    plt.ylabel('Daily Historical Demand (Units)')
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plot1_path = os.path.join(plots_dir, "lstm_actual_vs_predicted.png")
    plt.savefig(plot1_path, dpi=200)
    plt.close()
    
    # Plot 2: Training vs Validation Loss
    plt.figure(figsize=(8, 4.5))
    plt.plot(hist_dict['loss'], label='Training Loss (MSE)', color='#1976D2')
    plt.plot(hist_dict['val_loss'], label='Validation Loss (MSE)', color='#F57C00')
    plt.title('LSTM Training & Validation Convergence Curve', fontsize=12, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Squared Error')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plot2_path = os.path.join(plots_dir, "lstm_training_loss.png")
    plt.savefig(plot2_path, dpi=200)
    plt.close()
    
    # Plot 3: Residual / Error Distribution
    residuals = y_test_actual - y_pred_actual
    plt.figure(figsize=(8, 4.5))
    plt.hist(residuals, bins=30, color='#388E3C', edgecolor='black', alpha=0.7)
    plt.axvline(0, color='red', linestyle='dashed', linewidth=1.5)
    plt.title(f'LSTM Forecast Residual Distribution (Mean: {np.mean(residuals):.2f}, Std: {np.std(residuals):.2f})', fontsize=12, fontweight='bold')
    plt.xlabel('Prediction Residual (Actual - Predicted Demand)')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plot3_path = os.path.join(plots_dir, "lstm_residuals.png")
    plt.savefig(plot3_path, dpi=200)
    plt.close()
    
    print(f"Generated and saved 3 diagnostic plots to: {plots_dir}", flush=True)
    return metrics

if __name__ == "__main__":
    train_lstm_pipeline()
