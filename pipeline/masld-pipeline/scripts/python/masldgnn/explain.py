# -*- coding: utf-8 -*-
"""explain.py — GNNExplainer wrapper for drug-target link explanation.

Refactored from the last definition in 09_graphsage_pipeline.py
(lines 3295+).  Replaces `display()` with file outputs.
"""

import os
import numpy as np
import torch
import pandas as pd
import networkx as nx
from torch_geometric.explain import Explainer, GNNExplainer


@torch.no_grad()
def explain_drug_target_link(model, data, drug_idx, target_idx,
                                int_to_label, G_full_nx, save_dir="results"):
    """Run GNNExplainer on a single drug-target link prediction.

    Saves:
      - {drug}_{target}_subgraph_visualization.png
      - {drug}_{target}_subgraph_nodes.csv
      - {drug}_{target}_subgraph_edges.csv
      - {drug}_{target}_subgraph.sif
    """
    os.makedirs(save_dir, exist_ok=True)
    device = next(model.parameters()).device

    drug_label = int_to_label.get(drug_idx, f"node_{drug_idx}")
    target_label = int_to_label.get(target_idx, f"node_{target_idx}")
    safe_name = f"{drug_label}_{target_label}".replace(" ", "_")
    print(f"\nExplaining link: {drug_label} → {target_label}")

    # Setup explainer
    explainer = Explainer(
        model=model,
        algorithm=GNNExplainer(epochs=200, lr=0.01),
        explanation_type='model',
        node_mask_type='object',
        edge_mask_type='object',
        model_config=dict(
            mode='binary_classification',
            task_level='edge',
            return_type='raw',
        ),
    )

    # Build the edge label index for this single link
    edge_label_index = torch.tensor(
        [[drug_idx], [target_idx]], dtype=torch.long
    ).to(device)

    explanation = explainer(
        x=data.x.to(device),
        edge_index=data.edge_index.to(device),
        edge_label_index=edge_label_index,
    )

    node_mask = explanation.node_mask.cpu().numpy()
    edge_mask = explanation.edge_mask.cpu().numpy()

    # Filter to important nodes (top 20)
    node_importance = node_mask[:, 0] if node_mask.ndim > 1 else node_mask
    top_node_indices = np.argsort(node_importance)[-20:][::-1]
    top_node_indices = top_node_indices[node_importance[top_node_indices] > 0.01]

    # Build subgraph of important nodes + the drug/target pair
    sub_nodes = set(top_node_indices.tolist()) | {drug_idx, target_idx}
    edge_index_np = data.edge_index.cpu().numpy()
    sub_edges = []
    for i in range(edge_index_np.shape[1]):
        u, v = edge_index_np[:, i]
        if u in sub_nodes and v in sub_nodes:
            sub_edges.append((u, v, float(edge_mask[i])))

    # --- Save nodes CSV ---
    node_rows = []
    for idx in sorted(sub_nodes):
        label = int_to_label.get(idx, f"node_{idx}")
        node_type = G_full_nx.nodes[label].get('type', 'unknown') if label in G_full_nx else 'unknown'
        node_rows.append({
            'NodeID': idx,
            'Label': label,
            'Type': node_type,
            'Importance': float(node_importance[idx]),
        })
    nodes_df = pd.DataFrame(node_rows)
    nodes_csv = os.path.join(save_dir, f"{safe_name}_subgraph_nodes.csv")
    nodes_df.to_csv(nodes_csv, index=False)
    print(f"  Saved nodes → {nodes_csv}")

    # --- Save edges CSV ---
    edge_rows = []
    for u, v, w in sub_edges:
        edge_rows.append({
            'Source': int_to_label.get(u, f"node_{u}"),
            'Target': int_to_label.get(v, f"node_{v}"),
            'Weight': w,
        })
    edges_df = pd.DataFrame(edge_rows)
    edges_csv = os.path.join(save_dir, f"{safe_name}_subgraph_edges.csv")
    edges_df.to_csv(edges_csv, index=False)
    print(f"  Saved edges → {edges_csv}")

    # --- Save SIF ---
    sif_path = os.path.join(save_dir, f"{safe_name}_subgraph.sif")
    with open(sif_path, 'w') as f:
        for u, v, w in sub_edges:
            f.write(f"{int_to_label.get(u, u)} interacts {int_to_label.get(v, v)}\n")
    print(f"  Saved SIF  → {sif_path}")

    # --- Visualize ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        G_sub = nx.DiGraph()
        for idx in sub_nodes:
            label = int_to_label.get(idx, str(idx))
            G_sub.add_node(label, importance=float(node_importance[idx]))
        for u, v, w in sub_edges:
            ul = int_to_label.get(u, str(u))
            vl = int_to_label.get(v, str(v))
            G_sub.add_edge(ul, vl, weight=w)

        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        pos = nx.spring_layout(G_sub, seed=42, k=2.0)
        node_importances = [G_sub.nodes[n]['importance'] for n in G_sub.nodes()]
        nx.draw_networkx_nodes(G_sub, pos, ax=ax,
                               node_size=[200 + 2000 * imp for imp in node_importances],
                               node_color=node_importances, cmap='YlOrRd',
                               alpha=0.9, edgecolors='black')
        nx.draw_networkx_labels(G_sub, pos, ax=ax, font_size=8)
        edge_weights = [G_sub.edges[e]['weight'] for e in G_sub.edges()]
        nx.draw_networkx_edges(G_sub, pos, ax=ax,
                               width=[1 + 3 * w for w in edge_weights],
                               alpha=0.6, edge_color='gray',
                               arrows=True, arrowsize=15)
        ax.set_title(f"GNN Explainer: {drug_label} → {target_label}", fontsize=14, fontweight='bold')
        ax.axis('off')
        viz_path = os.path.join(save_dir, f"{safe_name}_subgraph_visualization.png")
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved viz   → {viz_path}")
    except Exception as e:
        print(f"  Visualization failed: {e}")

    return nodes_df, edges_df
