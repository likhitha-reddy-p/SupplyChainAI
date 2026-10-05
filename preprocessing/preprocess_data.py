"""
Data Cleaning and Chronological Train/Test Split Module
Author: Lead AI/ML Developer
Project: AI-Based Supply Chain Disruption Prediction and Recovery System
"""

import os
import pandas as pd
import numpy as np

def preprocess_and_split():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    excel_path = os.path.join(base_dir, "data", "integrated_synthetic_supply_chain_dataset.xlsx")
    processed_dir = os.path.join(base_dir, "data", "processed")
    os.makedirs(processed_dir, exist_ok=True)
    
    cleaned_csv_path = os.path.join(processed_dir, "cleaned_supply_chain.csv")
    train_csv_path = os.path.join(processed_dir, "train_2024.csv")
    test_csv_path = os.path.join(processed_dir, "test_2025.csv")
    
    print(f"Loading raw dataset from: {excel_path}...", flush=True)
    df = pd.read_excel(excel_path, sheet_name="Integrated_Data")
    print(f"Initial shape: {df.shape}", flush=True)
    
    # 1. Uniqueness check
    assert df['Record_ID'].nunique() == len(df), "Record_ID is not unique!"
    assert df['Shipment_ID'].nunique() == len(df), "Shipment_ID is not unique!"
    
    # 2. Check and remove exact duplicates
    dups = df.duplicated().sum()
    if dups > 0:
        print(f"Removing {dups} exact duplicate rows...", flush=True)
        df = df.drop_duplicates().reset_index(drop=True)
    else:
        print("No exact duplicate rows found.", flush=True)
        
    # 3. Check duplicate columns
    # Delay_Days vs Delivery_Delay_Days
    if 'Delay_Days' in df.columns and 'Delivery_Delay_Days' in df.columns:
        diff_delay = (df['Delay_Days'] - df['Delivery_Delay_Days']).abs().max()
        if diff_delay == 0:
            print("Verified: Delay_Days is identical to Delivery_Delay_Days. Dropping Delay_Days.", flush=True)
            df = df.drop(columns=['Delay_Days'])
            
    # Production_Loss_Pct vs Production_Impact_Pct
    if 'Production_Loss_Pct' in df.columns and 'Production_Impact_Pct' in df.columns:
        diff_prod = (df['Production_Loss_Pct'] - df['Production_Impact_Pct']).abs().max()
        if diff_prod == 0:
            print("Verified: Production_Loss_Pct is identical to Production_Impact_Pct. Dropping Production_Loss_Pct.", flush=True)
            df = df.drop(columns=['Production_Loss_Pct'])
            
    # 4. Check constant columns
    if 'Currency' in df.columns and df['Currency'].nunique() <= 1:
        print("Verified: Currency is constant ('USD'). Dropping Currency.", flush=True)
        df = df.drop(columns=['Currency'])
        
    # 5. Impute Disruption_Severity
    if 'Disruption_Severity' in df.columns:
        null_sev = df['Disruption_Severity'].isnull().sum()
        print(f"Imputing {null_sev:,} missing Disruption_Severity values to 'No Disruption'...", flush=True)
        df['Disruption_Severity'] = df['Disruption_Severity'].fillna("No Disruption")
        
    # 6. Normalize string columns (strip whitespace, ensure proper string type)
    str_cols = df.select_dtypes(include=['object']).columns
    for c in str_cols:
        if c != 'Date':
            df[c] = df[c].astype(str).str.strip()
            
    # 7. Convert Date and sort chronologically
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(by=['Date', 'Record_ID']).reset_index(drop=True)
    
    print(f"Cleaned dataset shape: {df.shape}", flush=True)
    print(f"Saving cleaned dataset to: {cleaned_csv_path}...", flush=True)
    df.to_csv(cleaned_csv_path, index=False)
    
    # 8. Temporal Train/Test Split: Strict 2024 vs 2025
    print("\n--- Performing Strict Temporal Train/Test Split ---", flush=True)
    df_train = df[df['Date'].dt.year == 2024].copy().reset_index(drop=True)
    df_test = df[df['Date'].dt.year == 2025].copy().reset_index(drop=True)
    
    print(f"Train 2024 rows: {len(df_train):,} ({df_train['Date'].min().strftime('%Y-%m-%d')} to {df_train['Date'].max().strftime('%Y-%m-%d')})", flush=True)
    print(f"Test 2025 rows: {len(df_test):,} ({df_test['Date'].min().strftime('%Y-%m-%d')} to {df_test['Date'].max().strftime('%Y-%m-%d')})", flush=True)
    
    # Validations
    assert len(df_train) + len(df_test) == len(df), "Train + Test record count does not equal total cleaned records!"
    assert (df_train['Date'].dt.year == 2025).sum() == 0, "CRITICAL ERROR: 2025 records leaked into training set!"
    assert (df_test['Date'].dt.year == 2024).sum() == 0, "CRITICAL ERROR: 2024 records present in test set!"
    assert df_train['Date'].is_monotonic_increasing, "Training set is not chronologically ordered!"
    assert df_test['Date'].is_monotonic_increasing, "Test set is not chronologically ordered!"
    
    print("Saving train_2024.csv...", flush=True)
    df_train.to_csv(train_csv_path, index=False)
    print("Saving test_2025.csv...", flush=True)
    df_test.to_csv(test_csv_path, index=False)
    
    print("\n[SUCCESS] Preprocessing and temporal split completed cleanly without error.", flush=True)

if __name__ == "__main__":
    preprocess_and_split()
