"""
Dataset Inspection and Analysis Module
Author: Lead AI/ML Developer
Project: AI-Based Supply Chain Disruption Prediction and Recovery System
"""

import os
import sys
import pandas as pd
import numpy as np

def run_dataset_analysis():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    excel_path = os.path.join(base_dir, "data", "integrated_synthetic_supply_chain_dataset.xlsx")
    report_path = os.path.join(base_dir, "preprocessing", "dataset_analysis_report.txt")
    
    print(f"Loading workbook from: {excel_path}", flush=True)
    if not os.path.exists(excel_path):
        raise FileNotFoundError(f"Original Excel dataset not found at: {excel_path}")
        
    xl = pd.ExcelFile(excel_path)
    sheet_names = xl.sheet_names
    print(f"Sheets found: {sheet_names}", flush=True)
    
    # Load metadata sheets
    df_dict = pd.read_excel(excel_path, sheet_name="Data_Dictionary")
    df_map = pd.read_excel(excel_path, sheet_name="Model_Mapping")
    
    # Load main dataset
    print("Loading Integrated_Data sheet (100,000 rows)...", flush=True)
    df = pd.read_excel(excel_path, sheet_name="Integrated_Data")
    
    lines = []
    lines.append("=" * 80)
    lines.append("SUPPLY CHAIN AI SYSTEM - DATASET INSPECTION & ANALYSIS REPORT")
    lines.append("=" * 80)
    lines.append(f"Source file: {excel_path}")
    lines.append(f"Workbook sheets: {sheet_names}")
    lines.append(f"Integrated_Data Dimensions: {df.shape[0]:,} rows × {df.shape[1]} columns")
    lines.append(f"Data_Dictionary Entries: {df_dict.shape[0]} rows")
    lines.append(f"Model_Mapping Entries: {df_map.shape[0]} rows")
    lines.append("-" * 80)
    
    # 1. Temporal Analysis
    df['Date_parsed'] = pd.to_datetime(df['Date'])
    min_date = df['Date_parsed'].min().strftime('%Y-%m-%d')
    max_date = df['Date_parsed'].max().strftime('%Y-%m-%d')
    unique_dates = df['Date_parsed'].nunique()
    year_counts = df['Date_parsed'].dt.year.value_counts().sort_index().to_dict()
    
    lines.append("\n1. TEMPORAL COVERAGE & CHRONOLOGICAL STRUCTURE:")
    lines.append(f"  - Minimum Date: {min_date}")
    lines.append(f"  - Maximum Date: {max_date}")
    lines.append(f"  - Total Unique Dates: {unique_dates} continuous calendar days")
    lines.append(f"  - Year 2024 Records (Train candidate): {year_counts.get(2024, 0):,}")
    lines.append(f"  - Year 2025 Records (Test candidate): {year_counts.get(2025, 0):,}")
    
    # 2. Uniqueness & Missing Values
    dup_rows = df.drop(columns=['Date_parsed']).duplicated().sum()
    rec_unique = df['Record_ID'].nunique()
    shp_unique = df['Shipment_ID'].nunique()
    
    lines.append("\n2. UNIQUENESS & INTEGRITY:")
    lines.append(f"  - Exact Duplicate Rows: {dup_rows}")
    lines.append(f"  - Unique Record_IDs: {rec_unique:,} / {len(df):,}")
    lines.append(f"  - Unique Shipment_IDs: {shp_unique:,} / {len(df):,}")
    
    null_counts = df.drop(columns=['Date_parsed']).isnull().sum()
    null_cols = null_counts[null_counts > 0]
    lines.append("\n3. MISSING VALUE AUDIT:")
    if len(null_cols) == 0:
        lines.append("  - No missing values across all columns.")
    else:
        for col, count in null_cols.items():
            pct = (count / len(df)) * 100
            lines.append(f"  - Column '{col}': {count:,} missing values ({pct:.2f}%)")
            # Check semantic connection with Disruption_Occurred
            if col == 'Disruption_Severity':
                disrupted_nulls = df[df['Disruption_Occurred'] == 1]['Disruption_Severity'].isnull().sum()
                non_disrupted_nulls = df[df['Disruption_Occurred'] == 0]['Disruption_Severity'].isnull().sum()
                lines.append(f"    * Non-disrupted records with null Severity: {non_disrupted_nulls:,} / {(df['Disruption_Occurred'] == 0).sum():,}")
                lines.append(f"    * Disrupted records with null Severity: {disrupted_nulls:,} / {(df['Disruption_Occurred'] == 1).sum():,}")
                lines.append(f"    * Semantic Justification: Null in Disruption_Severity represents non-disruption. Will impute with 'No Disruption'.")
    
    # 4. Redundant & Constant Columns
    lines.append("\n4. CONSTANT & REDUNDANT COLUMNS:")
    constant_cols = [c for c in df.columns if c != 'Date_parsed' and df[c].nunique() <= 1]
    for c in constant_cols:
        lines.append(f"  - Constant column detected: '{c}' with single value '{df[c].iloc[0]}'. Recommendation: Drop in preprocessing.")
        
    if 'Production_Loss_Pct' in df.columns and 'Production_Impact_Pct' in df.columns:
        diff1 = (df['Production_Loss_Pct'] - df['Production_Impact_Pct']).abs().max()
        corr1 = df[['Production_Loss_Pct', 'Production_Impact_Pct']].corr().iloc[0, 1]
        lines.append(f"  - Column redundancy: 'Production_Loss_Pct' vs 'Production_Impact_Pct' (Max Diff: {diff1}, Pearson Corr: {corr1:.4f}). Recommendation: Drop 'Production_Loss_Pct'.")
        
    if 'Delay_Days' in df.columns and 'Delivery_Delay_Days' in df.columns:
        diff2 = (df['Delay_Days'] - df['Delivery_Delay_Days']).abs().max()
        corr2 = df[['Delay_Days', 'Delivery_Delay_Days']].corr().iloc[0, 1]
        lines.append(f"  - Column redundancy: 'Delay_Days' vs 'Delivery_Delay_Days' (Max Diff: {diff2}, Pearson Corr: {corr2:.4f}). Recommendation: Drop 'Delay_Days'.")
    
    # 5. Target Variables & Post-Disruption Variables Audit
    lines.append("\n5. TARGETS, LEAKAGE CANDIDATES & POST-DISRUPTION AUDIT:")
    disr_counts = df['Disruption_Occurred'].value_counts().to_dict()
    disr_rate = df['Disruption_Occurred'].mean()
    lines.append(f"  - Primary Target 'Disruption_Occurred':")
    lines.append(f"    * Class 0 (No Disruption): {disr_counts.get(0, 0):,} ({((1 - disr_rate) * 100):.2f}%)")
    lines.append(f"    * Class 1 (Disruption): {disr_counts.get(1, 0):,} ({(disr_rate * 100):.2f}%)")
    lines.append(f"    * Imbalance Ratio (0:1): {(disr_counts.get(0, 0) / disr_counts.get(1, 1)):.2f} : 1")
    
    # Check Disruption_Probability
    if 'Disruption_Probability' in df.columns:
        corr_prob = df[['Disruption_Probability', 'Disruption_Occurred']].corr().iloc[0, 1]
        lines.append(f"  - Variable 'Disruption_Probability': Precomputed risk score. Corr with target = {corr_prob:.4f}. Must be EXCLUDED from pre-disruption features to prevent artificial target leakage.")
        
    post_event_cols = [
        'Disruption_Type', 'Disruption_Severity', 'Cargo_Condition', 'Order_Fulfillment_Status',
        'Delivery_Time_Deviation', 'Delivery_Delay_Days', 'Delay_Days', 'Inventory_Shortage_Units',
        'Financial_Loss_USD', 'Production_Impact_Pct', 'Production_Loss_Pct', 'Customer_Impact_Score',
        'Overall_Impact_Score', 'Recovery_Strategy', 'Recovery_Cost_USD', 'Recovery_Time_Days'
    ]
    lines.append("  - Post-Disruption Consequence Variables (Strictly excluded from pre-disruption XGBoost model):")
    for pec in post_event_cols:
        if pec in df.columns:
            lines.append(f"    * {pec}: Post-event consequence or recovery metric (Used only in Impact, Monte Carlo, or Optimization).")

    # 6. Entity Cardinality & Graph Structure
    lines.append("\n6. ENTITY CARDINALITIES & SUPPLY CHAIN GRAPH NETWORK:")
    entities = [
        ('Supplier_ID', 'Suppliers'),
        ('Factory_ID', 'Factories'),
        ('Origin_Port', 'Origin Ports'),
        ('Destination_Port', 'Destination Ports'),
        ('Warehouse_ID', 'Warehouses'),
        ('Route_ID', 'Routes'),
        ('Carrier', 'Carriers'),
        ('Product_ID', 'Products / SKUs'),
        ('Product_Category', 'Product Categories'),
        ('Transport_Mode', 'Transport Modes')
    ]
    for col, label in entities:
        if col in df.columns:
            lines.append(f"  - {label} ({col}): {df[col].nunique():,} unique values")
            
    # Write report
    report_text = "\n".join(lines)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)
        
    print(f"\nReport successfully saved to: {report_path}", flush=True)
    print("=" * 80)
    print(report_text[:1200] + "\n...[truncated for display]...")
    print("=" * 80)
    return report_path

if __name__ == "__main__":
    run_dataset_analysis()
