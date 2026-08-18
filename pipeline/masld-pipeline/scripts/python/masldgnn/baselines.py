# -*- coding: utf-8 -*-
"""baselines.py — Graph baselines (GCN, GAT, node2vec, XGBoost, heuristics).

Drop-in comparisons for the GNN link-prediction task.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.utils import to_undirected
from sklearn.metrics import roc_auc_score, average_precision_score
from collections import defaultdict
from node2vec import Node2Vec

from masldgnn.sampling import compute_pos_weight
from masldgnn.train import train_one_epoch, evaluate as gnn_evaluate


class GCN_LinkPredictor(nn.Module):
    """2-layer GCN + dot-product decoder (matches GraphSAGE API)."""
    def __init__(self, in_ch, hid, out):
        super().__init__()
        self.conv1 = GCNConv(in_ch, hid)
        self.conv2 = GCNConv(hid, out)

    def encode(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.conv2(x, edge_index)
        return x

    def decode(self, z, edge_label_index):
        src, dst = edge_label_index
        return (z[src] * z[dst]).sum(dim=-1)


class GAT_LinkPredictor(nn.Module):
    """2-layer GAT (1 head) + dot-product decoder."""
    def __init__(self, in_ch, hid, out):
        super().__init__()
        self.conv1 = GATConv(in_ch, hid, heads=1, concat=False)
        self.conv2 = GATConv(hid, out, heads=1, concat=False)

    def encode(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.conv2(x, edge_index)
        return x

    def decode(self, z, edge_label_index):
        src, dst = edge_label_index
        return (z[src] * z[dst]).sum(dim=-1)


def train_baseline_model(model, train_data, val_data, device,
                          num_epochs=200, patience=20, lr=0.005):
    """Generic training loop for any model with .encode/.decode."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    pw = compute_pos_weight(train_data.edge_label).to(device)
    best_auc, best_state, counter = 0.0, None, 0
    for epoch in range(1, num_epochs + 1):
        train_one_epoch(model, train_data, optimizer, device, pw)
        if epoch % 20 == 0:
            _, _, auc, _, _ = gnn_evaluate(model, val_data, device)
            if auc > best_auc:
                best_auc, best_state, counter = auc, model.state_dict(), 0
            else:
                counter += 1
            if counter >= patience:
                break
    if best_state:
        model.load_state_dict(best_state)
    return model


def run_gnn_baseline(model_cls, name, train_data, val_data, test_data,
                       in_dim, hid, out, device, **kwargs):
    """Run a GNN baseline and return test metrics dict."""
    print(f"\n--- Baseline: {name} ---")
    model = model_cls(in_dim, hid, out).to(device)
    model = train_baseline_model(model, train_data, val_data, device, **kwargs)
    acc, ap, auc, _, _ = gnn_evaluate(model, test_data, device)
    print(f"  {name}: Acc={acc:.3f}  AP={ap:.3f}  AUC={auc:.3f}")
    return {"baseline": name, "acc": acc, "ap": ap, "auc": auc}


def run_node2vec_baseline(G_nx, train_edges, train_labels,
                            test_edges, test_labels, node_map_int,
                            embedding_dim=64):
    """node2vec + logistic regression on Hadamard edge features."""
    from sklearn.linear_model import LogisticRegression

    print("\n--- Baseline: node2vec + LR ---")
    # Convert to simple undirected graph for node2vec
    G_simple = nx.Graph()
    for u, v in G_nx.edges():
        G_simple.add_edge(u, v)

    try:
        n2v = Node2Vec(G_simple, dimensions=embedding_dim, walk_length=30,
                       num_walks=10, workers=1, quiet=True)
        n2v_model = n2v.fit(window=5, min_count=1)
    except Exception as e:
        print(f"  node2vec failed: {e}")
        return {"baseline": "node2vec", "acc": 0.5, "ap": 0.5, "auc": 0.5}

    # Map integer node ids to node2vec embeddings
    int_to_label = {i: lbl for lbl, i in node_map_int.items()}
    n_nodes = len(int_to_label)
    emb_matrix = np.zeros((n_nodes, embedding_dim))
    found = 0
    for idx, lbl in int_to_label.items():
        if lbl in n2v_model.wv:
            emb_matrix[idx] = n2v_model.wv[lbl]
            found += 1
    print(f"  node2vec: {found}/{n_nodes} nodes mapped")

    # Edge features: Hadamard product
    def edge_features(pairs):
        feats = []
        for u, v in pairs:
            feats.append(emb_matrix[u] * emb_matrix[v])
        return np.array(feats)

    X_train = edge_features(train_edges.t().tolist())
    X_test = edge_features(test_edges.t().tolist())

    clf = LogisticRegression(max_iter=500)
    clf.fit(X_train, train_labels.cpu().numpy())
    preds = clf.predict_proba(X_test)[:, 1]
    labels = test_labels.cpu().numpy()

    if len(np.unique(labels)) > 1:
        auc = roc_auc_score(labels, preds)
        ap = average_precision_score(labels, preds)
        acc = ((preds > 0.5).astype(int) == labels).mean()
    else:
        auc = ap = acc = 0.5

    print(f"  node2vec+LR: Acc={acc:.3f}  AP={ap:.3f}  AUC={auc:.3f}")
    return {"baseline": "node2vec+LR", "acc": acc, "ap": ap, "auc": auc}


def run_xgboost_baseline(data, train_edges, train_labels,
                           test_edges, test_labels):
    """XGBoost on Hadamard product of node features (no GNN)."""
    from xgboost import XGBClassifier

    print("\n--- Baseline: XGBoost (Hadamard features) ---")
    u, v = train_edges
    X_train = data.x[u].numpy() * data.x[v].numpy()
    u, v = test_edges
    X_test = data.x[u].numpy() * data.x[v].numpy()

    clf = XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1,
                        use_label_encoder=False, eval_metric='logloss')
    clf.fit(X_train, train_labels.numpy())
    preds = clf.predict_proba(X_test)[:, 1]
    labels = test_labels.numpy()

    if len(np.unique(labels)) > 1:
        auc = roc_auc_score(labels, preds)
        ap = average_precision_score(labels, preds)
        acc = ((preds > 0.5).astype(int) == labels).mean()
    else:
        auc = ap = acc = 0.5

    print(f"  XGBoost: Acc={acc:.3f}  AP={ap:.3f}  AUC={auc:.3f}")
    return {"baseline": "XGBoost", "acc": acc, "ap": ap, "auc": auc}


def run_heuristic_baselines(G_nx, node_map_int, test_edges, test_labels):
    """Common-neighbour and Adamic-Adar heuristic baselines."""
    from sklearn.metrics import roc_auc_score, average_precision_score

    int_to_label = {i: lbl for lbl, i in node_map_int.items()}
    label_to_int = {lbl: i for lbl, i in node_map_int.items()}

    # Build integer-indexed adjacency
    adj = defaultdict(set)
    for u, v in G_nx.edges():
        if u in label_to_int and v in label_to_int:
            adj[label_to_int[u]].add(label_to_int[v])
            adj[label_to_int[v]].add(label_to_int[u])

    results = {}
    for name, func in [("Common Neighbours", lambda a, b: len(adj[a] & adj[b])),
                       ("Adamic-Adar", _adamic_adar)]:
        scores = []
        for u, v in test_edges.t().tolist():
            scores.append(func(u, v))
        scores = np.array(scores, dtype=float)
        if scores.max() > scores.min():
            scores = (scores - scores.min()) / (scores.max() - scores.min())
        labels = test_labels.numpy()
        if len(np.unique(labels)) > 1:
            auc = roc_auc_score(labels, scores)
            ap = average_precision_score(labels, scores)
        else:
            auc = ap = 0.5
        results[name] = {"baseline": name, "auc": auc, "ap": ap, "acc": 0.0}
        print(f"  {name}: AUC={auc:.3f}  AP={ap:.3f}")

    return results


def _adamic_adar(u, v, adj):
    aa = 0.0
    for common in adj[u] & adj[v]:
        deg = len(adj[common])
        if deg > 1:
            aa += 1.0 / np.log(deg)
    return aa
