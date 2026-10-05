"""
SHAP Explainability Module for XGBoost Disruption Predictions
Author: Lead AI/ML Developer
Project: AI-Based Supply Chain Disruption Prediction and Recovery System
"""

import os
import json
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

class DisruptionExplainer:
    def __init__(self):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.base_dir = base_dir
        self.model_path = os.path.join(base_dir, "models", "xgboost_disruption_model.json")
        self.feat_cols_path = os.path.join(base_dir, "data", "processed", "disruption_feature_columns.json")
        self.enc_path = os.path.join(base_dir, "data", "processed", "encoding_mappings.json")
        self.nodes_path = os.path.join(base_dir, "data", "processed", "nodes.csv")
        self.outputs_dir = os.path.join(base_dir, "explainability", "shap_outputs")
        os.makedirs(self.outputs_dir, exist_ok=True)
        
        # Load artifacts
        with open(self.feat_cols_path, "r") as f:
            self.feature_names = json.load(f)
        with open(self.enc_path, "r") as f:
            self.encodings = json.load(f)
            
        self.model = xgb.XGBClassifier()
        self.model.load_model(self.model_path)
        
        # Load GNN node metrics for enrichment
        if os.path.exists(self.nodes_path):
            df_nodes = pd.read_csv(self.nodes_path)
            self.node_vulnerability = dict(zip(df_nodes['node_id'], df_nodes['gnn_vulnerability_score']))
            self.node_betweenness = dict(zip(df_nodes['node_id'], df_nodes['betweenness_centrality']))
            self.node_pagerank = dict(zip(df_nodes['node_id'], df_nodes['pagerank']))
        else:
            self.node_vulnerability, self.node_betweenness, self.node_pagerank = {}, {}, {}
            
        # Initialize TreeExplainer
        self.explainer = shap.TreeExplainer(self.model)

    def prepare_feature_row(self, row_dict):
        """Converts raw shipment dictionary into exact model feature vector."""
        df_row = pd.DataFrame([row_dict])
        
        # Add GNN features
        sup_id = str(row_dict.get('Supplier_ID', ''))
        fac_id = str(row_dict.get('Factory_ID', ''))
        op_id = str(row_dict.get('Origin_Port', ''))
        wrh_id = str(row_dict.get('Warehouse_ID', ''))
        
        sup_risk = self.node_vulnerability.get(sup_id, 0.0)
        fac_risk = self.node_vulnerability.get(fac_id, 0.0)
        op_betw = self.node_betweenness.get(op_id, 0.0)
        wrh_pr = self.node_pagerank.get(wrh_id, 0.0)
        net_vuln = (sup_risk + fac_risk) / 2.0
        
        df_row['Supplier_GNN_Risk'] = sup_risk
        df_row['Factory_GNN_Risk'] = fac_risk
        df_row['Origin_Port_Betweenness'] = op_betw
        df_row['Warehouse_PageRank'] = wrh_pr
        df_row['Network_Vulnerability_Index'] = net_vuln
        
        # Apply frequency encodings
        for col, freq_map in self.encodings.get('frequency_encodings', {}).items():
            val = str(row_dict.get(col, ''))
            df_row[col + '_freq'] = freq_map.get(val, 0.0)
            
        # One-hot features
        vector_dict = {}
        for feat in self.feature_names:
            if feat in df_row.columns:
                try:
                    vector_dict[feat] = float(df_row[feat].iloc[0])
                except:
                    vector_dict[feat] = 0.0
            elif '_' in feat:
                # Check if it is a one-hot column
                prefix, val = feat.rsplit('_', 1)
                curr_val = str(row_dict.get(prefix, ''))
                vector_dict[feat] = 1.0 if curr_val == val else 0.0
            else:
                vector_dict[feat] = 0.0
                
        feat_df = pd.DataFrame([vector_dict])[self.feature_names].astype(np.float32)
        return feat_df

    def explain_instance(self, row_dict):
        """Calculates local SHAP values and returns structured human-readable drivers."""
        feat_df = self.prepare_feature_row(row_dict)
        shap_values = self.explainer(feat_df)
        
        values = shap_values.values[0]
        base_value = float(shap_values.base_values[0]) if hasattr(shap_values, 'base_values') else 0.0
        
        features_impact = []
        for feat, sh_val in zip(self.feature_names, values):
            val = feat_df[feat].iloc[0]
            # Human friendly naming
            clean_name = feat.replace('_freq', ' (Frequency Score)').replace('_', ' ')
            features_impact.append({
                'feature': feat,
                'readable_name': clean_name,
                'feature_value': float(val),
                'shap_value': float(sh_val),
                'direction': 'INCREASES RISK' if sh_val > 0 else 'REDUCES RISK'
            })
            
        # Sort by absolute contribution
        features_impact.sort(key=lambda x: abs(x['shap_value']), reverse=True)
        
        top_positive = [f for f in features_impact if f['shap_value'] > 0][:5]
        top_negative = [f for f in features_impact if f['shap_value'] < 0][:5]
        
        return {
            'base_value': base_value,
            'top_risk_increasing_factors': top_positive,
            'top_risk_reducing_factors': top_negative,
            'all_impacts': features_impact[:15]
        }

    def generate_global_analysis(self, n_samples=500):
        """Runs SHAP on test set sample and generates global summary plots."""
        test_path = os.path.join(self.base_dir, "data", "processed", "test_2025.csv")
        print(f"Loading test set for global SHAP analysis (sampling {n_samples} rows)...", flush=True)
        df_test = pd.read_csv(test_path).sample(n_samples, random_state=42).reset_index(drop=True)
        
        feature_rows = []
        for _, row in df_test.iterrows():
            f_row = self.prepare_feature_row(row.to_dict())
            feature_rows.append(f_row)
            
        X_sample = pd.concat(feature_rows, axis=0).reset_index(drop=True)
        
        print("Computing TreeSHAP values for sample...", flush=True)
        shap_values = self.explainer(X_sample)
        
        # 1. Global Bar Plot
        plt.figure(figsize=(10, 8))
        shap.plots.bar(shap_values, max_display=15, show=False)
        plt.title("Global Feature Attribution (Mean |SHAP Value|)", fontsize=13, fontweight='bold')
        plt.tight_layout()
        bar_plot_path = os.path.join(self.outputs_dir, "shap_bar_plot.png")
        plt.savefig(bar_plot_path, dpi=200)
        plt.close()
        
        # 2. Beeswarm Summary Plot
        plt.figure(figsize=(10, 8))
        shap.plots.beeswarm(shap_values, max_display=15, show=False)
        plt.title("SHAP Beeswarm Summary Plot: Impact on Disruption Likelihood", fontsize=13, fontweight='bold')
        plt.tight_layout()
        summary_plot_path = os.path.join(self.outputs_dir, "shap_summary_plot.png")
        plt.savefig(summary_plot_path, dpi=200)
        plt.close()
        
        # 3. Save JSON Feature Rankings
        mean_abs_shap = np.mean(np.abs(shap_values.values), axis=0)
        rankings = sorted(zip(self.feature_names, mean_abs_shap), key=lambda x: x[1], reverse=True)
        rankings_dict = {f: float(s) for f, s in rankings[:25]}
        
        json_path = os.path.join(self.outputs_dir, "global_shap_importance.json")
        with open(json_path, "w") as f:
            json.dump(rankings_dict, f, indent=2)
            
        print(f"SHAP artifacts saved cleanly to: {self.outputs_dir}", flush=True)
        return bar_plot_path, summary_plot_path

if __name__ == "__main__":
    explainer = DisruptionExplainer()
    print("Generating Global SHAP Analysis...", flush=True)
    explainer.generate_global_analysis(n_samples=400)
    
    # Test on a single sample record
    test_path = os.path.join(explainer.base_dir, "data", "processed", "test_2025.csv")
    sample_row = pd.read_csv(test_path).iloc[0].to_dict()
    print(f"\nExplaining Single Shipment: {sample_row['Shipment_ID']}...", flush=True)
    explanation = explainer.explain_instance(sample_row)
    print("Top factors increasing disruption risk:")
    for factor in explanation['top_risk_increasing_factors']:
        print(f"  + {factor['readable_name']}: SHAP = +{factor['shap_value']:.4f} (Value: {factor['feature_value']:.2f})")
    print("Top factors reducing disruption risk:")
    for factor in explanation['top_risk_reducing_factors']:
        print(f"  - {factor['readable_name']}: SHAP = {factor['shap_value']:.4f} (Value: {factor['feature_value']:.2f})")
