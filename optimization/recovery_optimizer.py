"""
Recovery Decision Optimization Module
Author: Lead AI/ML Developer
Project: AI-Based Supply Chain Disruption Prediction and Recovery System
"""

import os
import json
import pandas as pd
import numpy as np

class RecoveryOptimizer:
    """
    Multi-Criteria Decision Analysis (MCDA) and Constrained Recovery Optimizer.
    Evaluates feasible recovery actions under operational constraints to find
    the recommended recovery scenario based on configurable managerial priorities.
    """
    def __init__(self, data_path=None):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if data_path is None:
            data_path = os.path.join(base_dir, "data", "processed", "cleaned_supply_chain.csv")
        self.data_path = data_path
        self.empirical_benchmarks = self._extract_empirical_benchmarks()
        
    def _extract_empirical_benchmarks(self):
        df = pd.read_csv(self.data_path)
        benchmarks = {}
        strategies = ['No Action', 'Alternate Supplier', 'Shipment Rerouting', 'Inventory Reallocation']
        for strat in strategies:
            sub = df[df['Recovery_Strategy'] == strat]
            if len(sub) == 0:
                sub = df[df['Disruption_Occurred'] == 1]
            benchmarks[strat] = {
                'avg_cost': float(sub['Recovery_Cost_USD'].mean()),
                'avg_time': float(sub['Recovery_Time_Days'].mean()),
                'avg_delay': float(sub['Delivery_Delay_Days'].mean()),
                'avg_loss': float(sub['Financial_Loss_USD'].mean()),
                'avg_shortage': float(sub['Inventory_Shortage_Units'].mean())
            }
        return benchmarks

    def evaluate_recovery_options(self, shipment_record, w_cost=0.40, w_time=0.30, w_shortage=0.30):
        """
        Evaluates recovery options for a specific shipment under its constraints.
        Returns candidate actions, feasibility status, trade-offs, and recommended action.
        """
        # Extract constraints from record
        has_backup_sup = bool(shipment_record.get('Backup_Supplier_Available', 1))
        alt_supplier_id = shipment_record.get('Alternative_Supplier_ID', 'N/A')
        can_reroute = bool(shipment_record.get('Rerouting_Available', 1))
        can_reallocate = bool(shipment_record.get('Inventory_Reallocation_Possible', 1))
        inv_level = float(shipment_record.get('Inventory_Level', 300))
        safety_stock = float(shipment_record.get('Safety_Stock', 100))
        shipment_qty = float(shipment_record.get('Shipment_Quantity', 3500))
        
        # Scaling factor based on shipment volume
        scale = max(0.2, min(shipment_qty / 3500.0, 3.0))
        
        candidates = [
            {
                'action_id': 'ACT-1',
                'action_name': 'Continue Current Supplier (Wait & Expedite)',
                'description': 'Maintain order with primary supplier; absorb delay and expedite port handling upon arrival.',
                'feasible': True,
                'constraint_notes': 'Always feasible; high delay exposure.',
                'raw_cost': (self.empirical_benchmarks['No Action']['avg_cost'] + 5000) * scale,
                'raw_time': self.empirical_benchmarks['No Action']['avg_time'] + 2.0,
                'raw_delay': self.empirical_benchmarks['No Action']['avg_delay'] + 8.0,
                'raw_shortage': max(0.0, (shipment_qty - (inv_level + safety_stock)) * 0.15)
            },
            {
                'action_id': 'ACT-2',
                'action_name': 'Switch to Alternative Supplier',
                'description': f"Shift purchase order to secondary approved vendor ({alt_supplier_id}).",
                'feasible': has_backup_sup,
                'constraint_notes': f"Backup supplier availability: {'Available' if has_backup_sup else 'UNAVAILABLE (Infeasible)'}",
                'raw_cost': self.empirical_benchmarks['Alternate Supplier']['avg_cost'] * scale,
                'raw_time': self.empirical_benchmarks['Alternate Supplier']['avg_time'],
                'raw_delay': self.empirical_benchmarks['Alternate Supplier']['avg_delay'] * 0.7,
                'raw_shortage': max(0.0, (shipment_qty - (inv_level + safety_stock)) * 0.05) if has_backup_sup else 999.0
            },
            {
                'action_id': 'ACT-3',
                'action_name': 'Shipment Rerouting',
                'description': 'Divert transit through uncongested secondary port / intermodal corridor.',
                'feasible': can_reroute,
                'constraint_notes': f"Route corridor bypass: {'Feasible' if can_reroute else 'UNFEASIBLE (No alternate route)'}",
                'raw_cost': self.empirical_benchmarks['Shipment Rerouting']['avg_cost'] * scale,
                'raw_time': self.empirical_benchmarks['Shipment Rerouting']['avg_time'],
                'raw_delay': self.empirical_benchmarks['Shipment Rerouting']['avg_delay'] * 0.6,
                'raw_shortage': max(0.0, (shipment_qty - (inv_level + safety_stock)) * 0.08) if can_reroute else 999.0
            },
            {
                'action_id': 'ACT-4',
                'action_name': 'Inventory Reallocation',
                'description': 'Transfer buffer safety stock from adjacent regional fulfillment hub.',
                'feasible': can_reallocate and (inv_level + safety_stock > 100),
                'constraint_notes': f"Buffer transfer: {'Feasible' if can_reallocate else 'UNFEASIBLE (Insufficient regional stock)'}",
                'raw_cost': self.empirical_benchmarks['Inventory Reallocation']['avg_cost'] * scale,
                'raw_time': self.empirical_benchmarks['Inventory Reallocation']['avg_time'],
                'raw_delay': self.empirical_benchmarks['Inventory Reallocation']['avg_delay'] * 0.4,
                'raw_shortage': max(0.0, (shipment_qty - (inv_level + safety_stock)) * 0.02) if can_reallocate else 999.0
            }
        ]
        
        # Normalization upper bounds for MCDA scoring
        max_cost = max(c['raw_cost'] for c in candidates) or 1.0
        max_time = max(c['raw_time'] for c in candidates) or 1.0
        max_shortage = max(c['raw_shortage'] for c in candidates if c['feasible']) or 1.0
        
        scored_actions = []
        for c in candidates:
            if not c['feasible']:
                score = 999.0 # Penalty for infeasibility
            else:
                norm_c = c['raw_cost'] / max_cost
                norm_t = c['raw_time'] / max_time
                norm_s = c['raw_shortage'] / max_shortage if max_shortage > 0 else 0.0
                score = (w_cost * norm_c) + (w_time * norm_t) + (w_shortage * norm_s)
                
            c_dict = dict(c)
            c_dict['objective_loss_score'] = float(score)
            c_dict['formatted_cost'] = f"${c['raw_cost']:,.2f}"
            c_dict['formatted_time'] = f"{c['raw_time']:.1f} days"
            c_dict['formatted_delay'] = f"{c['raw_delay']:.1f} days"
            c_dict['formatted_shortage'] = f"{c['raw_shortage']:.1f} units"
            scored_actions.append(c_dict)
            
        # Rank feasible actions
        feasible_actions = [a for a in scored_actions if a['feasible']]
        feasible_actions.sort(key=lambda x: x['objective_loss_score'])
        
        recommended = feasible_actions[0] if len(feasible_actions) > 0 else scored_actions[0]
        
        return {
            'recommended_action': recommended['action_name'],
            'recommended_action_id': recommended['action_id'],
            'rationale': f"Selected as the recommended recovery scenario based on the configured objective "
                         f"(weights: Cost={w_cost:.2f}, Time={w_time:.2f}, Inventory Shortage={w_shortage:.2f}) "
                         f"with minimum objective loss of {recommended['objective_loss_score']:.3f}.",
            'objective_weights': {'cost': w_cost, 'time': w_time, 'shortage': w_shortage},
            'all_actions': scored_actions
        }

def run_standalone_optimizer():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cleaned_path = os.path.join(base_dir, "data", "processed", "cleaned_supply_chain.csv")
    recovery_csv_path = os.path.join(base_dir, "data", "processed", "recovery_data.csv")
    
    print("Generating recovery_data.csv...", flush=True)
    df = pd.read_csv(cleaned_path)
    recovery_cols = [
        'Shipment_ID', 'Date', 'Supplier_ID', 'Route_ID', 'Shipment_Quantity',
        'Inventory_Level', 'Safety_Stock', 'Backup_Supplier_Available',
        'Alternative_Supplier_ID', 'Rerouting_Available', 'Inventory_Reallocation_Possible',
        'Recovery_Strategy', 'Recovery_Cost_USD', 'Recovery_Time_Days', 'Financial_Loss_USD'
    ]
    recovery_cols = [c for c in recovery_cols if c in df.columns]
    df_recovery = df[recovery_cols].copy()
    df_recovery.to_csv(recovery_csv_path, index=False)
    print(f"Saved recovery dataset ({len(df_recovery):,} rows) to: {recovery_csv_path}", flush=True)
    
    # Test Optimization with a Sample Disrupted Shipment
    opt = RecoveryOptimizer()
    sample_record = df_recovery[df_recovery['Backup_Supplier_Available'] == 1].iloc[0].to_dict()
    print(f"\n--- Testing Recovery Optimizer for Shipment: {sample_record['Shipment_ID']} ---", flush=True)
    
    result = opt.evaluate_recovery_options(sample_record, w_cost=0.4, w_time=0.3, w_shortage=0.3)
    print("Recommended Action:", result['recommended_action'])
    print("Rationale:", result['rationale'])
    print("\nAction Comparison Table:")
    for a in result['all_actions']:
        status = "FEASIBLE" if a['feasible'] else "INFEASIBLE"
        print(f"  [{status}] {a['action_name']:<40} | Cost: {a['formatted_cost']:<12} | Time: {a['formatted_time']:<10} | Shortage: {a['formatted_shortage']:<12} | Score: {a['objective_loss_score']:.3f}")
        
    return result

if __name__ == "__main__":
    run_standalone_optimizer()
