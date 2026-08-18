# -*- coding: utf-8 -*-
"""loco.py — Leave-One-Class-Out Cross-Validation.

Refactored from the last definition in 09_graphsage_pipeline.py
(lines 2816-2919) with the B5 edge-removal bug FIXED.

The bug: in an undirected graph, NetworkX yields each edge in an
arbitrary orientation.  The old code checked only `u_label` for
type=='drug', so ~half the holdout-class edges survived into the
message-passing graph.

Fix (Part B5): iterate over BOTH endpoints of each edge.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected

from masldgnn.model import GraphSAGE_LinkPredictor
from masldgnn.sampling import compute_pos_weight
from masldgnn.train import train_one_epoch, evaluate


def _drug_class_for_node(node_label: str, class_mapping: dict) -> str:
    """Look up pharmacological class via substring match."""
    for drug_name, pharm_class in class_mapping.items():
        if drug_name.lower() in node_label.lower():
            return pharm_class
    return "Other_or_Gene"


def leave_one_class_out_cv(base_pyg_data: Data,
                             real_edge_label_index: torch.Tensor,
                             real_edge_label: torch.Tensor,
                             real_drug_classes: np.ndarray,
                             G_test_nx,
                             class_mapping: dict,
                             device,
                             num_epochs: int = 200,
                             gnn_dims=None):
    """LOCO cross-validation with the B5 edge-removal fix.

    Parameters
    ----------
    base_pyg_data : PyG Data (full graph features + node_labels)
    real_edge_label_index : (2, N) tensor of labeled edges
    real_edge_label : (N,) tensor of 0/1 labels
    real_drug_classes : (N,) array of per-edge drug class strings
    G_test_nx : nx.Graph with *string* labels (used for edge filtering)
    class_mapping : dict  drug_name → pharm_class
    device : torch.device
    num_epochs : training epochs per fold
    gnn_dims : list[int]  hidden dims

    Returns
    -------
    fold_results : list[dict]
    """
    if gnn_dims is None:
        gnn_dims = [64, 32]

    unique_classes = np.unique(real_drug_classes)
    fold_results = []

    for i, holdout_class in enumerate(unique_classes):
        print(f"\n--- LOCO Fold {i+1}/{len(unique_classes)}: "
              f"Holding out '{holdout_class}' ---")

        holdout_idx = np.where(real_drug_classes == holdout_class)[0]
        train_idx = np.where(real_drug_classes != holdout_class)[0]

        if len(holdout_idx) == 0 or len(train_idx) == 0:
            print(f"  Skipping: no edges for class '{holdout_class}'.")
            continue

        cv_train_idx = real_edge_label_index[:, train_idx].to(device)
        cv_train_lbl = real_edge_label[train_idx].to(device)
        cv_test_idx = real_edge_label_index[:, holdout_idx].to(device)
        cv_test_lbl = real_edge_label[holdout_idx].to(device)

        # ---- Build message-passing graph: remove holdout-class drug edges ----
        G_mp = G_test_nx.copy()
        edges_to_remove = []

        for u_label, v_label in G_mp.edges():
            # *** B5 FIX: check BOTH endpoints ***
            for node_label in (u_label, v_label):
                if G_mp.nodes[node_label].get('type') == 'drug':
                    node_drug_class = _drug_class_for_node(node_label,
                                                           class_mapping)
                    if node_drug_class == holdout_class:
                        edges_to_remove.append((u_label, v_label))
                        break  # no need to check the other endpoint

        G_mp.remove_edges_from(edges_to_remove)
        print(f"  Removed {len(edges_to_remove)} edges for class "
              f"'{holdout_class}'")

        # Convert filtered graph to edge_index
        label_to_idx = {lbl: idx
                        for idx, lbl in enumerate(base_pyg_data.node_labels)}
        mp_edges = []
        for u_lbl, v_lbl in G_mp.edges():
            if u_lbl in label_to_idx and v_lbl in label_to_idx:
                mp_edges.append((label_to_idx[u_lbl], label_to_idx[v_lbl]))

        if mp_edges:
            mp_ei = torch.tensor(mp_edges, dtype=torch.long).t().contiguous()
        else:
            mp_ei = torch.empty((2, 0), dtype=torch.long)
        mp_ei = to_undirected(mp_ei, num_nodes=base_pyg_data.num_nodes).to(device)

        # Build fold training data
        fold_data = Data(x=base_pyg_data.x.to(device), edge_index=mp_ei)
        fold_data.edge_label_index = cv_train_idx
        fold_data.edge_label = cv_train_lbl

        # Train
        in_dim = base_pyg_data.x.size(1)
        model = GraphSAGE_LinkPredictor(in_dim, gnn_dims[0], gnn_dims[1]).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.005,
                                       weight_decay=5e-4)
        pw = compute_pos_weight(fold_data.edge_label).to(device)

        n_test_pos = int(cv_test_lbl.sum().item())
        print(f"  Training: {fold_data.edge_label_index.size(1)} links, "
              f"Test: {len(holdout_idx)} ({n_test_pos} pos)")

        for epoch in range(1, num_epochs + 1):
            train_one_epoch(model, fold_data, optimizer, device, pw)
            if epoch % 50 == 0:
                _, _, auc_tr, _, _ = evaluate(model, fold_data, device)
                print(f"    Epoch {epoch:03d} | Train AUC: {auc_tr:.3f}")

        # Evaluate on held-out class
        acc, ap, roc_auc, _, _ = evaluate(
            model, fold_data, device,
            test_edge_label_index=cv_test_idx,
            test_edge_label=cv_test_lbl,
        )

        fold_results.append({
            "holdout_class": holdout_class,
            "n_test": len(holdout_idx),
            "n_test_pos": n_test_pos,
            "accuracy": float(acc),
            "ap": float(ap),
            "roc_auc": float(roc_auc),
        })
        print(f"  Fold '{holdout_class}': Acc={acc:.3f}  AP={ap:.3f}  "
              f"AUC={roc_auc:.3f}  (n={len(holdout_idx)})")

    # Summary
    if fold_results:
        df = pd.DataFrame(fold_results)
        print(f"\n{'='*60}")
        print("LOCO CV Summary")
        print(df.round(3).to_string(index=False))
        print(f"\n  Mean AUC: {df['roc_auc'].mean():.3f} +/- {df['roc_auc'].std():.3f}")
        print(f"  Mean AP:  {df['ap'].mean():.3f} +/- {df['ap'].std():.3f}")
        print(f"{'='*60}")
    else:
        print("No LOCO results.")
        df = pd.DataFrame()

    return df, fold_results
