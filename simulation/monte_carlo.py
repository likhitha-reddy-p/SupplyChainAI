"""
Monte Carlo Simulation Module for Disruption and Recovery Scenarios
Author: Lead AI/ML Developer
Project: AI-Based Supply Chain Disruption Prediction and Recovery System
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

np.random.seed(42)

class SupplyChainMonteCarloSimulator:
    def __init__(self, data_path=None):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if data_path is None:
            data_path = os.path.join(base_dir, "data", "processed", "cleaned_supply_chain.csv")
        self.data_path = data_path
        self.empirical_params = {}
        self.fit_empirical_distributions()
        
    def fit_empirical_distributions(self):
        """Fit empirical distribution parameters from actual historical data by strategy."""
        df = pd.read_csv(self.data_path)
        
        # Mapping standard scenario names to dataset strategy categories
        scenario_mapping = {
            'Scenario A (Current Supplier)': 'No Action',
            'Scenario B (Alternative Supplier)': 'Alternate Supplier',
            'Scenario C (Shipment Rerouting)': 'Shipment Rerouting',
            'Scenario D (Inventory Reallocation)': 'Inventory Reallocation'
        }
        
        metrics = ['Recovery_Cost_USD', 'Recovery_Time_Days', 'Delivery_Delay_Days', 'Financial_Loss_USD', 'Inventory_Shortage_Units']
        
        for sc_name, strat_val in scenario_mapping.items():
            sub = df[df['Recovery_Strategy'] == strat_val]
            if len(sub) == 0:
                sub = df[df['Disruption_Occurred'] == 1]
                
            self.empirical_params[sc_name] = {}
            for m in metrics:
                vals = sub[m].dropna().values
                mean_val = float(np.mean(vals)) if len(vals) > 0 else 1.0
                std_val = float(np.std(vals)) if len(vals) > 0 else 0.5
                min_val = float(np.min(vals)) if len(vals) > 0 else 0.0
                q95 = float(np.percentile(vals, 95)) if len(vals) > 0 else mean_val
                
                self.empirical_params[sc_name][m] = {
                    'mean': mean_val,
                    'std': max(std_val, 0.1),
                    'min': min_val,
                    'p95': q95
                }
                
    def run_simulation(self, n_iterations=2000, shipment_context=None):
        """
        Runs Monte Carlo iterations for each of the 4 recovery scenarios.
        Optional shipment_context can scale the financial and delay outcomes
        based on shipment distance and value.
        """
        results = {}
        
        scale_factor_financial = 1.0
        scale_factor_delay = 1.0
        
        if shipment_context is not None:
            # Scale proportionally to shipment quantity and distance if provided
            qty = shipment_context.get('Shipment_Quantity', 3500)
            dist = shipment_context.get('Distance_km', 10000)
            scale_factor_financial = (qty / 3500.0) * (dist / 10000.0)
            scale_factor_financial = max(0.2, min(scale_factor_financial, 3.0))
            
            lead = shipment_context.get('Lead_Time_Days', 20)
            scale_factor_delay = lead / 20.0
            scale_factor_delay = max(0.5, min(scale_factor_delay, 2.0))
            
        for sc_name, params in self.empirical_params.items():
            # Generate parametric samples with log-normal or truncated normal distributions
            # 1. Recovery Cost
            cost_p = params['Recovery_Cost_USD']
            cost_sim = np.maximum(0, np.random.normal(cost_p['mean'], cost_p['std'], n_iterations))
            
            # 2. Recovery Time (Days)
            time_p = params['Recovery_Time_Days']
            time_sim = np.maximum(0, np.random.normal(time_p['mean'], time_p['std'], n_iterations))
            
            # 3. Delivery Delay (Days)
            delay_p = params['Delivery_Delay_Days']
            delay_sim = np.maximum(0, np.random.normal(delay_p['mean'] * scale_factor_delay, delay_p['std'], n_iterations))
            
            # 4. Inventory Shortage (Units)
            shortage_p = params['Inventory_Shortage_Units']
            shortage_sim = np.maximum(0, np.random.normal(shortage_p['mean'], shortage_p['std'], n_iterations))
            
            # 5. Financial Loss (USD)
            loss_p = params['Financial_Loss_USD']
            loss_sim = np.maximum(0, np.random.normal(loss_p['mean'] * scale_factor_financial, loss_p['std'], n_iterations))
            
            # Total Cost Impact = Direct Financial Loss + Recovery Cost
            total_financial_sim = cost_sim + loss_sim
            
            # Composite Scenario Outcome Risk Score (0 - 100)
            # Normalized combination of delay, financial impact, and recovery duration
            norm_delay = np.clip(delay_sim / 30.0, 0, 1)
            norm_loss = np.clip(total_financial_sim / 500000.0, 0, 1)
            norm_time = np.clip(time_sim / 25.0, 0, 1)
            composite_score = (0.4 * norm_loss + 0.35 * norm_delay + 0.25 * norm_time) * 100.0
            
            results[sc_name] = {
                'samples': {
                    'recovery_cost': cost_sim.tolist(),
                    'recovery_time': time_sim.tolist(),
                    'delivery_delay': delay_sim.tolist(),
                    'inventory_shortage': shortage_sim.tolist(),
                    'financial_loss': total_financial_sim.tolist(),
                    'composite_outcome_score': composite_score.tolist()
                },
                'summary': {
                    'expected_recovery_cost': float(np.mean(cost_sim)),
                    'expected_recovery_time_days': float(np.mean(time_sim)),
                    'expected_delay_days': float(np.mean(delay_sim)),
                    'expected_inventory_shortage': float(np.mean(shortage_sim)),
                    'expected_total_financial_loss': float(np.mean(total_financial_sim)),
                    'expected_outcome_risk_score': float(np.mean(composite_score)),
                    'p95_financial_loss_var': float(np.percentile(total_financial_sim, 95)),
                    'p95_delay_days': float(np.percentile(delay_sim, 95))
                }
            }
            
        return results

def run_standalone_simulation():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_dir = os.path.join(base_dir, "simulation", "simulation_results")
    os.makedirs(results_dir, exist_ok=True)
    
    summary_path = os.path.join(results_dir, "monte_carlo_summary.json")
    plot_path = os.path.join(results_dir, "monte_carlo_scenarios_comparison.png")
    
    print("Initializing Monte Carlo Supply Chain Simulator...", flush=True)
    sim = SupplyChainMonteCarloSimulator()
    
    print("Executing 2,000 simulation iterations across 4 recovery scenarios...", flush=True)
    res = sim.run_simulation(n_iterations=2000)
    
    # Save JSON summary (without sample arrays for compactness)
    summaries = {sc: data['summary'] for sc, data in res.items()}
    with open(summary_path, "w") as f:
        json.dump(summaries, f, indent=2)
    print(f"Saved simulation summary to: {summary_path}", flush=True)
    
    # Print formatted summary table
    print("\n" + "=" * 95)
    print(f"{'Recovery Scenario':<36} | {'Exp Delay (d)':<14} | {'Exp Cost ($)':<14} | {'Exp Shortage':<12} | {'Risk Score'}")
    print("-" * 95)
    for sc, s in summaries.items():
        print(f"{sc:<36} | {s['expected_delay_days']:<14.1f} | ${s['expected_total_financial_loss']:<13,.0f} | {s['expected_inventory_shortage']:<12.1f} | {s['expected_outcome_risk_score']:.1f}/100")
    print("=" * 95)
    
    # Generate Comparative Distribution Plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    scenarios = list(res.keys())
    colors = ['#E53935', '#FB8C00', '#1E88E5', '#43A047']
    
    # Subplot 1: Total Financial Loss Distribution
    ax1 = axes[0, 0]
    for sc, col in zip(scenarios, colors):
        losses = np.array(res[sc]['samples']['financial_loss']) / 1000.0
        ax1.hist(losses, bins=35, alpha=0.5, label=sc, color=col)
    ax1.set_title('Total Financial Impact Distribution (Loss + Cost)', fontweight='bold')
    ax1.set_xlabel('Financial Impact ($ Thousands)')
    ax1.set_ylabel('Simulated Frequency')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # Subplot 2: Delivery Delay Distribution
    ax2 = axes[0, 1]
    for sc, col in zip(scenarios, colors):
        delays = res[sc]['samples']['delivery_delay']
        ax2.hist(delays, bins=35, alpha=0.5, label=sc, color=col)
    ax2.set_title('Delivery Delay Distribution (Days)', fontweight='bold')
    ax2.set_xlabel('Effective Delay (Days)')
    ax2.set_ylabel('Simulated Frequency')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    # Subplot 3: Recovery Time Distribution
    ax3 = axes[1, 0]
    for sc, col in zip(scenarios, colors):
        times = res[sc]['samples']['recovery_time']
        ax3.hist(times, bins=35, alpha=0.5, label=sc, color=col)
    ax3.set_title('Recovery Duration Distribution (Days)', fontweight='bold')
    ax3.set_xlabel('Recovery Time (Days)')
    ax3.set_ylabel('Simulated Frequency')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)
    
    # Subplot 4: Composite Outcome Risk Score
    ax4 = axes[1, 1]
    for sc, col in zip(scenarios, colors):
        scores = res[sc]['samples']['composite_outcome_score']
        ax4.hist(scores, bins=35, alpha=0.5, label=sc, color=col)
    ax4.set_title('Composite Outcome Risk Score (0 - 100)', fontweight='bold')
    ax4.set_xlabel('Scenario Risk Score')
    ax4.set_ylabel('Simulated Frequency')
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle('Monte Carlo Supply Chain Disruption & Recovery Simulation (2,000 Iterations)', fontsize=14, fontweight='bold', y=0.99)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"Saved simulation distribution comparison plot to: {plot_path}", flush=True)

if __name__ == "__main__":
    run_standalone_simulation()
