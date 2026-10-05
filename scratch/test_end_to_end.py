"""
End-to-End System Integration Test
Author: Lead AI/ML Developer
Project: AI-Based Supply Chain Disruption Prediction and Recovery System
"""

import os
import sys
import pandas as pd
import numpy as np

# Ensure project root is in sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from qr_barcode.shipment_identifier import ShipmentQRIdentifier
from simulation.monte_carlo import SupplyChainMonteCarloSimulator
from optimization.recovery_optimizer import RecoveryOptimizer
from explainability.shap_analysis import DisruptionExplainer
import joblib

def run_e2e_test():
    print("=" * 80)
    print("STARTING END-TO-END PIPELINE INTEGRATION TEST")
    print("=" * 80)
    
    # 1. Initialize modules
    print("\n[Step 1] Initializing system modules...", flush=True)
    qr = ShipmentQRIdentifier()
    explainer = DisruptionExplainer()
    simulator = SupplyChainMonteCarloSimulator()
    optimizer = RecoveryOptimizer()
    
    impact_path = os.path.join(BASE_DIR, "models", "impact_models.pkl")
    impact_data = joblib.load(impact_path)
    impact_models = impact_data['models']
    
    # 2. Select 4 diverse test shipments
    test_shipment_ids = [
        "SHP-0000001", # Benchmark 2024 record
        "SHP-0050214", # First 2025 record
        "SHP-0050215", # Second 2025 record
        "SHP-0075000"  # Mid 2025 record
    ]
    
    print(f"\n[Step 2] Testing across {len(test_shipment_ids)} distinct shipment records...", flush=True)
    
    for shp_id in test_shipment_ids:
        print("\n" + "-" * 75)
        print(f"TESTING PIPELINE FOR SHIPMENT: {shp_id}")
        print("-" * 75)
        
        # A. QR Generation & Lookup
        rec = qr.lookup_shipment(shp_id)
        assert rec is not None, f"Failed to retrieve record for {shp_id}"
        print(f"A. Ingestion & Telemetry:")
        print(f"   Route: {rec.get('Route_ID')} | Mode: {rec.get('Transport_Mode')} | Category: {rec.get('Product_Category')}")
        print(f"   Supplier: {rec.get('Supplier_ID')} | Country: {rec.get('Supplier_Country')} | Distance: {rec.get('Distance_km')} km")
        print(f"   Shipment Quantity: {rec.get('Shipment_Quantity'):,} units | Lead Time: {rec.get('Lead_Time_Days'):.1f} days")
        
        # B. Disruption Risk Scoring (XGBoost)
        feat_df = explainer.prepare_feature_row(rec)
        prob = float(explainer.model.predict_proba(feat_df)[0, 1])
        tier = "HIGH RISK" if prob > 0.65 else ("MODERATE RISK" if prob >= 0.35 else "LOW RISK")
        print(f"\nB. Disruption Risk Prediction (XGBoost):")
        print(f"   Predicted Disruption Probability: {prob*100:.1f}% -> Tier: [{tier}]")
        
        # C. Post-Disruption Consequence Impact Analysis
        row_df = pd.DataFrame([rec])
        X_num = row_df[impact_data['feature_cols']].copy()
        X_cat = pd.get_dummies(row_df[impact_data['cat_cols']], drop_first=False)
        X_cat = X_cat.reindex(columns=impact_data['cat_encoded_cols'], fill_value=0)
        X_imp = pd.concat([X_num, X_cat], axis=1).astype(np.float32)
        
        est_delay = float(impact_models['Delivery_Delay_Days'].predict(X_imp)[0])
        est_shortage = float(impact_models['Inventory_Shortage_Units'].predict(X_imp)[0])
        est_loss = float(impact_models['Financial_Loss_USD'].predict(X_imp)[0])
        est_score = float(impact_models['Overall_Impact_Score'].predict(X_imp)[0])
        
        print(f"\nD. Consequence Impact Estimation:")
        print(f"   Expected Delay: {est_delay:.1f} days | Inventory Shortage: {est_shortage:.0f} units")
        print(f"   Projected Loss: ${est_loss:,.0f} | Overall Impact Score: {est_score:.1f}/100")
        
        # D. Monte Carlo Recovery Simulation (2,000 runs)
        sim_res = simulator.run_simulation(n_iterations=1000, shipment_context=rec)
        print(f"\nE. Monte Carlo Scenario Simulation (1,000 iterations):")
        for sc_name, sc_info in sim_res.items():
            s = sc_info['summary']
            print(f"   * {sc_name:<36}: Exp Delay = {s['expected_delay_days']:.1f}d | Total Loss = ${s['expected_total_financial_loss']:<10,.0f} | Risk = {s['expected_outcome_risk_score']:.1f}")
            
        # E. Recovery Action Optimization (MCDA)
        opt_res = optimizer.evaluate_recovery_options(rec, w_cost=0.40, w_time=0.30, w_shortage=0.30)
        print(f"\nF. Recovery Optimization:")
        print(f"   Recommended Scenario: {opt_res['recommended_action']}")
        print(f"   Rationale: {opt_res['rationale']}")
        
        # F. Explainability (TreeSHAP)
        explanation = explainer.explain_instance(rec)
        top_pos = explanation['top_risk_increasing_factors']
        top_neg = explanation['top_risk_reducing_factors']
        print(f"\nG. SHAP Feature Attribution:")
        if top_pos:
            print(f"   Top Risk Driver (+): {top_pos[0]['readable_name']} (Value: {top_pos[0]['feature_value']:.2f}, SHAP: +{top_pos[0]['shap_value']:.4f})")
        if top_neg:
            print(f"   Top Protective Factor (-): {top_neg[0]['readable_name']} (Value: {top_neg[0]['feature_value']:.2f}, SHAP: {top_neg[0]['shap_value']:.4f})")
            
        print(f"\n>>> PIPELINE TEST PASSED FOR {shp_id} <<<")
        
    print("\n" + "=" * 80)
    print("ALL END-TO-END PIPELINE INTEGRATION TESTS SUCCEEDED!")
    print("=" * 80)

if __name__ == "__main__":
    run_e2e_test()
