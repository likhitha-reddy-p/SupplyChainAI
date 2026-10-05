"""
Supply Chain Graph Extraction and GNN Architecture Module
Author: Lead AI/ML Developer
Project: AI-Based Supply Chain Disruption Prediction and Recovery System
"""

import os
import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx

# Reproducibility seed
torch.manual_seed(42)
np.random.seed(42)

class SupplyChainGCN(nn.Module):
    """
    Two-layer Graph Convolutional Network for Supply Chain Risk Propagation.
    Propagates upstream supplier vulnerability and port transit risk across echelons.
    """
    def __init__(self, in_features, hidden_dim, out_dim, dropout=0.2):
        super(SupplyChainGCN, self).__init__()
        self.conv1 = nn.Linear(in_features, hidden_dim)
        self.conv2 = nn.Linear(hidden_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
        self.risk_head = nn.Linear(out_dim, 1)

    def forward(self, x, adj_norm):
        # First layer: message passing
        h1 = torch.spmm(adj_norm, x)
        h1 = F.relu(self.conv1(h1))
        h1 = self.dropout(h1)
        
        # Second layer: multi-hop propagation
        h2 = torch.spmm(adj_norm, h1)
        h2 = F.relu(self.conv2(h2))
        
        # Predict node vulnerability / risk level
        node_risk = torch.sigmoid(self.risk_head(h2))
        return h2, node_risk

def build_and_train_graph():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    train_path = os.path.join(base_dir, "data", "processed", "train_2024.csv")
    nodes_path = os.path.join(base_dir, "data", "processed", "nodes.csv")
    edges_path = os.path.join(base_dir, "data", "processed", "edges.csv")
    gnn_model_path = os.path.join(base_dir, "models", "gnn_network_model.pt")
    gnn_metrics_path = os.path.join(base_dir, "models", "gnn_metrics.json")
    
    print("Loading 2024 training data to construct supply chain topology...", flush=True)
    df_train = pd.read_csv(train_path)
    
    # 1. Extract Unique Nodes across 5 Echelons
    suppliers = sorted(df_train['Supplier_ID'].unique().tolist())
    factories = sorted(df_train['Factory_ID'].unique().tolist())
    origin_ports = sorted(df_train['Origin_Port'].unique().tolist())
    dest_ports = sorted(df_train['Destination_Port'].unique().tolist())
    warehouses = sorted(df_train['Warehouse_ID'].unique().tolist())
    
    all_nodes = []
    for s in suppliers:
        all_nodes.append({'node_id': s, 'node_type': 'Supplier', 'echelon': 1})
    for f in factories:
        all_nodes.append({'node_id': f, 'node_type': 'Factory', 'echelon': 2})
    for op in origin_ports:
        all_nodes.append({'node_id': op, 'node_type': 'Origin_Port', 'echelon': 3})
    for dp in dest_ports:
        all_nodes.append({'node_id': dp, 'node_type': 'Destination_Port', 'echelon': 4})
    for w in warehouses:
        all_nodes.append({'node_id': w, 'node_type': 'Warehouse', 'echelon': 5})
        
    df_nodes = pd.DataFrame(all_nodes)
    node_id_to_idx = {row['node_id']: idx for idx, row in df_nodes.iterrows()}
    num_nodes = len(df_nodes)
    print(f"Constructed {num_nodes} nodes across 5 echelons: "
          f"{len(suppliers)} Suppliers, {len(factories)} Factories, {len(origin_ports)} Origin Ports, "
          f"{len(dest_ports)} Dest Ports, {len(warehouses)} Warehouses.", flush=True)
    
    # 2. Extract Edges supported directly by the dataset
    edges_list = []
    
    # Echelon 1 -> 2: Supplier -> Factory
    e1 = df_train[['Supplier_ID', 'Factory_ID']].drop_duplicates()
    for _, row in e1.iterrows():
        edges_list.append({'source': row['Supplier_ID'], 'target': row['Factory_ID'], 'edge_type': 'SUPPLIER_TO_FACTORY'})
        
    # Echelon 2 -> 3: Factory -> Origin_Port
    e2 = df_train[['Factory_ID', 'Origin_Port']].drop_duplicates()
    for _, row in e2.iterrows():
        edges_list.append({'source': row['Factory_ID'], 'target': row['Origin_Port'], 'edge_type': 'FACTORY_TO_ORIGIN_PORT'})
        
    # Echelon 3 -> 4: Origin_Port -> Destination_Port
    e3 = df_train[['Origin_Port', 'Destination_Port']].drop_duplicates()
    for _, row in e3.iterrows():
        edges_list.append({'source': row['Origin_Port'], 'target': row['Destination_Port'], 'edge_type': 'ORIGIN_TO_DEST_PORT'})
        
    # Echelon 4 -> 5: Destination_Port -> Warehouse
    e4 = df_train[['Destination_Port', 'Warehouse_ID']].drop_duplicates()
    for _, row in e4.iterrows():
        edges_list.append({'source': row['Destination_Port'], 'target': row['Warehouse_ID'], 'edge_type': 'DEST_PORT_TO_WAREHOUSE'})
        
    df_edges = pd.DataFrame(edges_list)
    print(f"Extracted {len(df_edges)} unique directed edges from training set.", flush=True)
    
    # 3. Calculate NetworkX Centrality Metrics
    G = nx.DiGraph()
    for _, row in df_nodes.iterrows():
        G.add_node(row['node_id'], node_type=row['node_type'])
    for _, row in df_edges.iterrows():
        G.add_edge(row['source'], row['target'], edge_type=row['edge_type'])
        
    in_degree = dict(G.in_degree())
    out_degree = dict(G.out_degree())
    betweenness = nx.betweenness_centrality(G)
    pagerank = nx.pagerank(G)
    
    df_nodes['in_degree'] = df_nodes['node_id'].map(in_degree)
    df_nodes['out_degree'] = df_nodes['node_id'].map(out_degree)
    df_nodes['betweenness_centrality'] = df_nodes['node_id'].map(betweenness)
    df_nodes['pagerank'] = df_nodes['node_id'].map(pagerank)
    
    # 4. Compute Node Historical Disruption Rates & Operational Attributes (Training Set ONLY)
    node_disr_rates = {}
    node_reliability = {}
    
    # Supplier metrics
    sup_grp = df_train.groupby('Supplier_ID')
    for s, g in sup_grp:
        node_disr_rates[s] = g['Disruption_Occurred'].mean()
        node_reliability[s] = g['Supplier_Reliability_Score'].mean()
        
    # Factory metrics
    fac_grp = df_train.groupby('Factory_ID')
    for f, g in fac_grp:
        node_disr_rates[f] = g['Disruption_Occurred'].mean()
        node_reliability[f] = g['Capacity_Utilization'].mean() / 100.0
        
    # Origin Port metrics
    op_grp = df_train.groupby('Origin_Port')
    for op, g in op_grp:
        node_disr_rates[op] = g['Disruption_Occurred'].mean()
        node_reliability[op] = g['Handling_Equipment_Availability'].mean() / 100.0
        
    # Destination Port metrics
    dp_grp = df_train.groupby('Destination_Port')
    for dp, g in dp_grp:
        node_disr_rates[dp] = g['Disruption_Occurred'].mean()
        node_reliability[dp] = 1.0 - (g['Customs_Clearance_Time'].mean() / 10.0)
        
    # Warehouse metrics
    wrh_grp = df_train.groupby('Warehouse_ID')
    for w, g in wrh_grp:
        node_disr_rates[w] = g['Disruption_Occurred'].mean()
        node_reliability[w] = g['Safety_Stock'].mean() / max(g['Inventory_Level'].mean(), 1.0)
        
    df_nodes['historical_disruption_rate'] = df_nodes['node_id'].map(node_disr_rates).fillna(0.0)
    df_nodes['operational_reliability'] = df_nodes['node_id'].map(node_reliability).fillna(0.5)
    
    df_nodes.to_csv(nodes_path, index=False)
    df_edges.to_csv(edges_path, index=False)
    print(f"Saved nodes to {nodes_path} and edges to {edges_path}.", flush=True)
    
    # 5. Build PyTorch Tensors for GNN
    # Features per node: [echelon_norm, in_deg, out_deg, betweenness, pagerank, reliability]
    feature_matrix = []
    labels = []
    for _, row in df_nodes.iterrows():
        feats = [
            row['echelon'] / 5.0,
            row['in_degree'] / 10.0,
            row['out_degree'] / 10.0,
            row['betweenness_centrality'],
            row['pagerank'] * 10.0,
            row['operational_reliability']
        ]
        feature_matrix.append(feats)
        labels.append(row['historical_disruption_rate'])
        
    X = torch.tensor(feature_matrix, dtype=torch.float32)
    Y = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)
    
    # Build Normalized Adjacency Matrix with Self-Loops (GCN normalization: D^-1/2 * (A + I) * D^-1/2)
    A = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    for _, row in df_edges.iterrows():
        u = node_id_to_idx[row['source']]
        v = node_id_to_idx[row['target']]
        A[u, v] = 1.0
        A[v, u] = 1.0 # Bidirectional propagation along supply lines
        
    A_tilde = A + np.eye(num_nodes, dtype=np.float32)
    degrees = np.sum(A_tilde, axis=1)
    d_inv_sqrt = np.power(degrees, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    D_tilde = np.diag(d_inv_sqrt)
    adj_norm_np = D_tilde.dot(A_tilde).dot(D_tilde)
    adj_norm = torch.tensor(adj_norm_np, dtype=torch.float32)
    
    # 6. Train SupplyChainGCN Model
    model = SupplyChainGCN(in_features=6, hidden_dim=16, out_dim=8, dropout=0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
    criterion = nn.MSELoss()
    
    model.train()
    print("Training Supply Chain GCN for 150 epochs...", flush=True)
    for epoch in range(150):
        optimizer.zero_grad()
        _, pred_risk = model(X, adj_norm)
        loss = criterion(pred_risk, Y)
        loss.backward()
        optimizer.step()
        
    model.eval()
    with torch.no_grad():
        embeddings, predicted_node_risk = model(X, adj_norm)
        final_mse = criterion(predicted_node_risk, Y).item()
        
    print(f"GNN Training Complete. Final Node Vulnerability MSE: {final_mse:.6f}", flush=True)
    
    # 7. Extract Node Risk Scores & Embeddings
    node_risk_map = {df_nodes.iloc[i]['node_id']: float(predicted_node_risk[i].item()) for i in range(num_nodes)}
    df_nodes['gnn_vulnerability_score'] = df_nodes['node_id'].map(node_risk_map)
    df_nodes.to_csv(nodes_path, index=False)
    
    # Save Model Checkpoint
    torch.save({
        'model_state_dict': model.state_dict(),
        'node_id_to_idx': node_id_to_idx,
        'adj_norm': adj_norm,
        'node_features': X,
        'num_nodes': num_nodes
    }, gnn_model_path)
    
    metrics = {
        'num_nodes': num_nodes,
        'num_edges': len(df_edges),
        'node_types': {
            'Suppliers': len(suppliers),
            'Factories': len(factories),
            'Origin_Ports': len(origin_ports),
            'Destination_Ports': len(dest_ports),
            'Warehouses': len(warehouses)
        },
        'edge_types': df_edges['edge_type'].value_counts().to_dict(),
        'gnn_architecture': '2-layer Graph Convolutional Network (GCN)',
        'final_node_vulnerability_mse': final_mse,
        'features_used': ['echelon_norm', 'in_degree', 'out_degree', 'betweenness_centrality', 'pagerank', 'operational_reliability']
    }
    
    with open(gnn_metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
        
    print(f"GNN artifacts saved to {gnn_model_path} and {gnn_metrics_path}.", flush=True)
    return metrics, node_risk_map

if __name__ == "__main__":
    build_and_train_graph()
