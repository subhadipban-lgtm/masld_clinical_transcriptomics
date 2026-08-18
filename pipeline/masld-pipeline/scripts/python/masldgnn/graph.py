# -*- coding: utf-8 -*-
from __future__ import annotations
"""graph.py — GEXF loading, PyG conversion, feature engineering.

Canonical (last) definition from 09_graphsage_pipeline.py, cleaned.
Returns (Data, G_mapped_nx, node_map_int) — the 4th/5th variant
that also returns the mapped graph and index dict.
"""

import numpy as np
import torch
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected


def scan_all_categories(gexf_paths: list[str]) -> tuple[set, set]:
    """Pre-scan one or more GEXF files to collect all categorical values.

    Returns (all_node_types, all_personalization_statuses).
    """
    all_node_types: set[str] = set()
    all_statuses: set[str] = set()
    for p in gexf_paths:
        G = nx.read_gexf(p)
        all_node_types |= set(nx.get_node_attributes(G, "type").values())
        all_statuses |= set(nx.get_node_attributes(G, "personalization_status").values())
    all_node_types.add("unknown")
    all_statuses.add("unknown")
    return all_node_types, all_statuses


def preprocess_graph_for_pyg(G_nx: nx.Graph,
                               all_node_types: set,
                               all_personalization_statuses: set):
    """Convert a NetworkX graph to a PyG Data object.

    Categorical attributes (type, personalization_status) are one-hot
    encoded.  Numeric attributes are clipped and stacked.

    Returns
    -------
    data : torch_geometric.data.Data
    G_mapped_nx : nx.Graph  (node labels replaced with integer IDs)
    node_map_int : dict[str, int]  (label → integer index)
    """
    print(f"Preprocessing graph: {G_nx.number_of_nodes()} nodes, "
          f"{G_nx.number_of_edges()} edges...")

    type_map = {name: i for i, name in enumerate(sorted(all_node_types))}
    status_map = {name: i for i, name in enumerate(sorted(all_personalization_statuses))}
    node_labels = sorted(G_nx.nodes())
    node_map_int = {label: i for i, label in enumerate(node_labels)}
    G_mapped_nx = nx.relabel_nodes(G_nx, node_map_int, copy=True)

    node_features = []
    for node_id in range(len(node_labels)):
        original_label = node_labels[node_id]
        attr = G_nx.nodes[original_label]

        # One-hot type
        type_vec = torch.zeros(len(type_map))
        t = attr.get("type", "unknown")
        if t in type_map:
            type_vec[type_map[t]] = 1.0

        # One-hot personalization_status
        status_vec = torch.zeros(len(status_map))
        s = attr.get("personalization_status", "unknown")
        if s in status_map:
            status_vec[status_map[s]] = 1.0

        # Numeric features
        logFC = float(attr.get("dge_logFC_Late_vs_Early", 0.0))
        pval = float(attr.get("dge_adj_p_val", 1.0))
        is_suppressor = float(bool(attr.get("is_suppressor", False)))
        in_fibrosis = float(bool(attr.get("in_fibrosis_module", False)))
        is_driver = float(bool(attr.get("is_driver", False)))

        pval_transformed = -np.log10(pval + 1e-12)
        pval_transformed = np.clip(pval_transformed, 0, 50)
        logFC = np.clip(logFC, -10, 10)

        numeric = torch.tensor(
            [logFC, pval_transformed, is_suppressor, in_fibrosis, is_driver],
            dtype=torch.float,
        )
        node_features.append(torch.cat([type_vec, status_vec, numeric]))

    x = torch.stack(node_features)
    edges = list(G_mapped_nx.edges())
    if edges:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)

    data = Data(
        x=x,
        edge_index=to_undirected(edge_index, num_nodes=len(node_labels)),
        node_labels=node_labels,
        num_nodes=len(node_labels),
    )

    print(f"  Nodes: {data.num_nodes}, Features per node: {x.size(1)}, "
          f"Edges: {data.num_edges}")
    return data, G_mapped_nx, node_map_int
