# -*- coding: utf-8 -*-
"""C7 — Embedding-Degeneracy Diagnostic

Do this FIRST — it may change your conclusions.

Usage:
    conda run -n masld-env python scripts/python/15_embedding_diagnostics.py \
        --gexf data/masld_personalized_kg_enhanced.gexf \
        --weights results/masld_personalized_kg_enhanced.pt

Reads a trained GraphSAGE model and its knowledge graph, computes
pairwise cosine similarity among drug embeddings, effective rank,
and per-drug graph degree.  Saves a histogram to results/.
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")  # headless on macOS / server
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F

# Ensure the masldgnn package is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from masldgnn.config import load_config, get_device
from masldgnn.graph import preprocess_graph_for_pyg, scan_all_categories
from masldgnn.model import GraphSAGE_LinkPredictor


def run_diagnostics(gexf_path: str, weights_path: str, save_dir: str):
    os.makedirs(save_dir, exist_ok=True)
    cfg = load_config()
    device = get_device()

    # ---- Load graph ----
    print(f"Loading graph from {gexf_path} …")
    G_nx = nx.read_gexf(gexf_path)
    all_node_types, all_statuses = scan_all_categories([gexf_path])
    data, _, node_map_int = preprocess_graph_for_pyg(G_nx, all_node_types, all_statuses)

    # ---- Load model ----
    in_dim = data.x.size(1)
    if os.path.isfile(weights_path):
        state_dict = torch.load(weights_path, map_location=device)
        # Determine layers and dims from state_dict
        if "conv3.lin_l.weight" in state_dict:
            h1 = state_dict["conv1.lin_l.weight"].shape[0]
            h2 = state_dict["conv2.lin_l.weight"].shape[0]
            out_ch = state_dict["conv3.lin_l.weight"].shape[0]
            model = GraphSAGE_LinkPredictor(in_dim, h1, out_ch, h2).to(device)
        elif "conv2.lin_l.weight" in state_dict:
            h1 = state_dict["conv1.lin_l.weight"].shape[0]
            out_ch = state_dict["conv2.lin_l.weight"].shape[0]
            model = GraphSAGE_LinkPredictor(in_dim, h1, out_ch).to(device)
        else:
            dims = cfg["GNN_DIMS"]
            if len(dims) == 3:
                model = GraphSAGE_LinkPredictor(in_dim, dims[0], dims[1], dims[2]).to(device)
            else:
                model = GraphSAGE_LinkPredictor(in_dim, dims[0], dims[1]).to(device)

        # strict=True: verify exact matching architecture
        model.load_state_dict(state_dict, strict=True)
        print(f"Loaded weights from {weights_path} with strict=True")
    else:
        print(f"WARNING: weights not found at {weights_path}. Using random init.")
        dims = cfg["GNN_DIMS"]
        if len(dims) == 3:
            model = GraphSAGE_LinkPredictor(in_dim, dims[0], dims[1], dims[2]).to(device)
        else:
            model = GraphSAGE_LinkPredictor(in_dim, dims[0], dims[1]).to(device)

    model.eval()

    # ---- Extract drug embeddings ----
    int_to_label = {i: lbl for lbl, i in node_map_int.items()}
    drug_indices = [
        i for lbl, i in node_map_int.items()
        if G_nx.nodes[lbl].get("type") == "drug"
    ]

    if not drug_indices:
        print("ERROR: No drug nodes found in graph.")
        sys.exit(1)

    with torch.no_grad():
        z_all = model.encode(data.x.to(device), data.edge_index.to(device))
    z_all = z_all.cpu()

    drug_z = z_all[drug_indices]  # (n_drugs, out_channels)
    drug_labels = [int_to_label[i] for i in drug_indices]

    # ---- Pairwise cosine similarity ----
    Zn = F.normalize(drug_z, dim=1)
    S = (Zn @ Zn.T).numpy()
    off_diag = S[~np.eye(len(S), dtype=bool)]

    print(f"\n{'='*60}")
    print(f"  EMBEDDING DIAGNOSTICS  ({len(drug_labels)} drug nodes)")
    print(f"{'='*60}")
    print(f"  mean pairwise cosine : {off_diag.mean():.4f}")
    print(f"  median pairwise cosine: {np.median(off_diag):.4f}")
    print(f"  fraction > 0.99       : {(off_diag > 0.99).mean():.4f}")
    print(f"  embedding norms  min  : {drug_z.norm(dim=1).min().item():.4f}")
    print(f"  embedding norms  med  : {drug_z.norm(dim=1).median().item():.4f}")
    print(f"  embedding norms  max  : {drug_z.norm(dim=1).max().item():.4f}")

    # ---- Effective rank (participation ratio) ----
    sv = torch.linalg.svdvals(drug_z - drug_z.mean(0))
    pr = (sv.sum() ** 2) / (sv ** 2).sum()
    out_dim = drug_z.size(1)
    print(f"  effective rank (of {out_dim}): {pr.item():.2f}")

    # ---- Per-drug degree ----
    print(f"\n  Per-drug degrees in G:")
    for d in ["Resmetirom", "Liraglutide", "Semaglutide",
              "Obeticholic acid", "Disulfiram", "Sulfinpyrazone",
              "Balsalazide", "Olsalazine"]:
        # fuzzy match
        matches = [n for n in G_nx.nodes() if d.lower() in n.lower()]
        if matches:
            print(f"    {matches[0]:40s}  degree = {G_nx.degree(matches[0])}")
        else:
            print(f"    {d:40s}  ABSENT")

    # ---- Interpretation ----
    print(f"\n{'='*60}")
    if off_diag.mean() > 0.9 or pr.item() < 3:
        print("  ⚠  WARNING: Embedding space appears COLLAPSED.")
        print("     Drug nodes likely lack informative features.")
        print("     Consider adding ATC one-hot, target count, Morgan PCAs.")
    else:
        print("  ✓  Embedding space appears well-differentiated.")
    print(f"{'='*60}")

    # ---- Histogram ----
    plt.figure(figsize=(8, 5))
    plt.hist(off_diag, bins=50, color="#457b9d", edgecolor="white")
    plt.xlabel("Pairwise Cosine Similarity")
    plt.ylabel("Count")
    plt.title("Drug Embedding Pairwise Cosine Similarity Distribution")
    plt.tight_layout()
    out_path = os.path.join(save_dir, "drug_similarity_distribution.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"\nSaved histogram → {out_path}")

    # ---- Save stats JSON ----
    import json
    stats = {
        "n_drugs": len(drug_labels),
        "mean_cosine": float(off_diag.mean()),
        "median_cosine": float(np.median(off_diag)),
        "frac_gt_099": float((off_diag > 0.99).mean()),
        "norm_min": float(drug_z.norm(dim=1).min().item()),
        "norm_median": float(drug_z.norm(dim=1).median().item()),
        "norm_max": float(drug_z.norm(dim=1).max().item()),
        "effective_rank": float(pr.item()),
        "embedding_dim": out_dim,
    }
    stats_path = os.path.join(save_dir, "stats_graph_audit.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Saved stats    → {stats_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="C7 Embedding Degeneracy Diagnostic")
    parser.add_argument("--gexf", required=True, help="Path to enhanced GEXF knowledge graph")
    parser.add_argument("--weights", required=True, help="Path to trained .pt model weights")
    parser.add_argument("--save-dir", default="results", help="Output directory")
    args = parser.parse_args()
    run_diagnostics(args.gexf, args.weights, args.save_dir)
