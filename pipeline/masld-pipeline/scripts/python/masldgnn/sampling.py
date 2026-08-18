# -*- coding: utf-8 -*-
"""sampling.py — Negative edge sampling and data splitting.

Canonical version (last definition, lines ~1688-1725 of the original),
with degree-preserving negative sampling added (Part C6).
"""

import random
from collections import defaultdict

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected


def _degree_preserving_negatives(pos_edge_set, int_to_type, nodes_by_type,
                                  edge_types_to_sample, target_count):
    """Sample negatives with probability proportional to node degree."""
    # Build degree dict
    all_nodes = list(int_to_type.keys())
    degree = defaultdict(int)
    for u, v in pos_edge_set:
        degree[u] += 1
        degree[v] += 1
    max_deg = max(degree.values()) if degree else 1

    # Weight proportional to degree
    weights = {n: degree.get(n, 1) / max_deg for n in all_nodes}
    total_w = sum(weights.values())
    probs = {n: w / total_w for n, w in weights.items()}
    node_list = list(probs.keys())
    prob_list = [probs[n] for n in node_list]

    neg_set = set()
    for _ in range(target_count * 20):
        if len(neg_set) >= target_count:
            break
        for edge_type, count in edge_types_to_sample.items():
            type_a, type_b = edge_type
            cands_a = [n for n in node_list if int_to_type.get(n) == type_a]
            cands_b = [n for n in node_list if int_to_type.get(n) == type_b]
            if not cands_a or not cands_b:
                continue
            # Weighted sample
            idx_a = np.random.choice(len(cands_a), p=[probs[n] for n in cands_a] / sum(probs[n] for n in cands_a))
            idx_b = np.random.choice(len(cands_b), p=[probs[n] for n in cands_b] / sum(probs[n] for n in cands_b))
            u, v = cands_a[idx_a], cands_b[idx_b]
            if u == v:
                continue
            pair = tuple(sorted((u, v)))
            if pair not in pos_edge_set and pair not in neg_set:
                neg_set.add(pair)
            if len(neg_set) >= target_count:
                break
    return neg_set


def create_hard_negative_splits(data: Data,
                                  G_nx_mapped,
                                  int_to_type: dict,
                                  val_size: float = 0.05,
                                  test_size: float = 0.1,
                                  balance_dataset: bool = False,
                                  degree_preserving: bool = False,
                                  use_2hop_sampling: bool = False):
    """Create train/val/test edge splits with negative sampling.

    Parameters
    ----------
    data : PyG Data with edge_index and node_labels
    G_nx_mapped : nx.Graph with integer node labels (from preprocess_graph_for_pyg)
    int_to_type : dict mapping int node id → node type string
    val_size, test_size : fraction of positive edges
    balance_dataset : downsample to 1:1 pos/neg
    degree_preserving : use degree-proportional negative sampling
    use_2hop_sampling : use 2-hop negative sampling (type-matched)
    """
    if balance_dataset:
        print("  - Strategy: 1:1 Balanced")
    if degree_preserving:
        print("  - Strategy: Degree-preserving negatives")

    nodes_by_type = defaultdict(list)
    for node_idx, n_type in int_to_type.items():
        nodes_by_type[n_type].append(node_idx)

    pos_edge_set = {tuple(sorted((u, v))) for u, v in data.edge_index.t().tolist()}

    edge_types_to_sample = defaultdict(int)
    for u, v in pos_edge_set:
        edge_types_to_sample[tuple(sorted((int_to_type[u], int_to_type[v])))] += 1

    neg_edge_set: set = set()
    target_neg_count = len(pos_edge_set)

    # --- 2-Hop Sampling ---
    if use_2hop_sampling:
        print("  - Strategy: 2-Hop Sampling")
        adj = defaultdict(set)
        for u, v in pos_edge_set:
            adj[u].add(v)
            adj[v].add(u)
        for u, v in list(pos_edge_set):
            if len(neg_edge_set) >= target_neg_count:
                break
            two_hop = set()
            for nbr in adj[u]:
                for two in adj[nbr]:
                    two_hop.add(two)
            cands = [w for w in two_hop
                     if w != u and w not in adj[u]
                     and int_to_type.get(w) == int_to_type.get(v)]
            if cands:
                w = random.choice(cands)
                pair = tuple(sorted((u, w)))
                if pair not in pos_edge_set:
                    neg_edge_set.add(pair)

    # --- Degree-preserving or Random fallback ---
    if len(neg_edge_set) < target_neg_count:
        remaining = target_neg_count - len(neg_edge_set)
        if degree_preserving:
            new_negs = _degree_preserving_negatives(
                pos_edge_set, int_to_type, nodes_by_type,
                edge_types_to_sample, remaining)
        else:
            new_negs = set()
            for edge_type, count in edge_types_to_sample.items():
                type_a, type_b = edge_type
                nodes_a, nodes_b = nodes_by_type[type_a], nodes_by_type[type_b]
                if not nodes_a or not nodes_b:
                    continue
                for _ in range(count * 10):
                    if len(neg_edge_set) + len(new_negs) >= target_neg_count:
                        break
                    u = random.choice(nodes_a)
                    v = random.choice(nodes_b)
                    if u == v:
                        continue
                    pair = tuple(sorted((u, v)))
                    if pair not in pos_edge_set and pair not in neg_edge_set and pair not in new_negs:
                        new_negs.add(pair)
        neg_edge_set |= new_negs

    print(f"  - Total negatives: {len(neg_edge_set)}")

    pos_edges_list = list(pos_edge_set)
    neg_edges_list = list(neg_edge_set)

    if balance_dataset:
        min_count = min(len(pos_edges_list), len(neg_edges_list))
        print(f"  - Downsampling to {min_count} pos / {min_count} neg")
        pos_edges_list = random.sample(pos_edges_list, min_count)
        neg_edges_list = random.sample(neg_edges_list, min_count)

    random.shuffle(pos_edges_list)
    random.shuffle(neg_edges_list)

    # --- Split ---
    val_count_pos = int(len(pos_edges_list) * val_size)
    test_count_pos = int(len(pos_edges_list) * test_size)
    train_pos = pos_edges_list[val_count_pos + test_count_pos:]
    val_pos = pos_edges_list[:val_count_pos]
    test_pos = pos_edges_list[val_count_pos:val_count_pos + test_count_pos]

    val_count_neg = int(len(neg_edges_list) * val_size)
    test_count_neg = int(len(neg_edges_list) * test_size)
    train_neg = neg_edges_list[val_count_neg + test_count_neg:]
    val_neg = neg_edges_list[:val_count_neg]
    test_neg = neg_edges_list[val_count_neg:val_count_neg + test_count_neg]

    def _make_split(pos_list, neg_list, train_edges_tensor):
        split = Data(
            x=data.x,
            edge_index=to_undirected(train_edges_tensor, num_nodes=data.num_nodes),
            node_labels=data.node_labels,
            num_nodes=data.num_nodes,
        )
        pos_t = torch.tensor(pos_list, dtype=torch.long).t().contiguous() if pos_list else torch.empty((2, 0), dtype=torch.long)
        neg_t = torch.tensor(neg_list, dtype=torch.long).t().contiguous() if neg_list else torch.empty((2, 0), dtype=torch.long)
        split.edge_label_index = torch.cat([pos_t, neg_t], dim=1)
        split.edge_label = torch.cat([
            torch.ones(pos_t.size(1)),
            torch.zeros(neg_t.size(1)),
        ]).float()
        return split

    train_pos_tensor = (torch.tensor(train_pos, dtype=torch.long).t().contiguous()
                        if train_pos else torch.empty((2, 0), dtype=torch.long))

    return (
        _make_split(train_pos, train_neg, train_pos_tensor),
        _make_split(val_pos, val_neg, train_pos_tensor),
        _make_split(test_pos, test_neg, train_pos_tensor),
    )


def compute_pos_weight(edge_label: torch.Tensor) -> torch.Tensor:
    """Weight for BCEWithLogitsLoss to handle class imbalance."""
    pos = (edge_label == 1).sum()
    neg = (edge_label == 0).sum()
    return neg / pos if pos > 0 and neg > 0 else torch.tensor(1.0)
