# -*- coding: utf-8 -*-
"""C2 — Clinical-Trial Benchmark

Loads data/trial_outcomes.csv, scores every drug via the trained GNN
(cosine similarity to the MASLD-Fibrosis disease node), and computes
AUROC for success-vs-failure separation with bootstrap CI.

Usage:
    conda run -n masld-env python scripts/python/11_trial_benchmark.py \
        --gexf data/masld_personalized_kg_enhanced.gexf \
        --weights results/masld_personalized_kg_enhanced.pt

Interpretation (decide BEFORE running):
    AUROC >= 0.70 → strong, quotable result
    AUROC ~= 0.5 → embedding does not encode therapeutic efficacy
"""

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from masldgnn.config import load_config, get_device
from masldgnn.graph import preprocess_graph_for_pyg, scan_all_categories
from masldgnn.model import GraphSAGE_LinkPredictor


TRIAL_OUTCOMES_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "trial_outcomes.csv"
)


def _fuzzy_match(drug_name: str, graph_nodes: list[str]) -> str | None:
    """Fuzzy-match a trial drug name to a KG node."""
    drug_lower = drug_name.lower().strip()
    for node in graph_nodes:
        if drug_lower in node.lower():
            return node
    return None


def run_benchmark(gexf_path: str, weights_path: str, save_dir: str):
    os.makedirs(save_dir, exist_ok=True)
    cfg = load_config()
    device = get_device()

    # Load trial outcomes
    trial_df = pd.read_csv(TRIAL_OUTCOMES_PATH)
    print(f"Loaded {len(trial_df)} trial entries.")
    print(trial_df.to_string(index=False))

    # Load graph and model
    print(f"\nLoading graph from {gexf_path} …")
    import networkx as nx
    G_nx = nx.read_gexf(gexf_path)
    all_types, all_statuses = scan_all_categories([gexf_path])
    data, _, node_map_int = preprocess_graph_for_pyg(G_nx, all_types, all_statuses)
    int_to_label = {i: lbl for lbl, i in node_map_int.items()}

    in_dim = data.x.size(1)
    dims = cfg["GNN_DIMS"]
    if len(dims) == 3:
        model = GraphSAGE_LinkPredictor(in_dim, dims[0], dims[1], dims[2]).to(device)
    else:
        model = GraphSAGE_LinkPredictor(in_dim, dims[0], dims[1]).to(device)

    if os.path.isfile(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=device), strict=False)
        print(f"Loaded weights from {weights_path}")
    else:
        print(f"WARNING: no weights at {weights_path}, using random init.")

    model.eval()

    # Get embeddings
    with torch.no_grad():
        z = model.encode(data.x.to(device), data.edge_index.to(device)).cpu()

    # Find disease node (fuzzy: 'MASLD' or 'fibrosis' or 'NASH')
    disease_idx = None
    for lbl, idx in node_map_int.items():
        if any(kw in lbl.lower() for kw in ['masld', 'nash', 'fibrosis', 'mash']):
            if G_nx.nodes[lbl].get('type') == 'disease':
                disease_idx = idx
                disease_label = lbl
                break
    if disease_idx is None:
        print("WARNING: no disease node found. Using mean gene embedding as proxy.")
        gene_idxs = [i for lbl, i in node_map_int.items()
                     if G_nx.nodes[lbl].get('type') == 'gene']
        disease_vec = z[gene_idxs].mean(dim=0)
    else:
        disease_vec = z[disease_idx]
        print(f"Using disease node: {disease_label}")

    # Score each drug by cosine similarity to disease node
    graph_nodes = list(G_nx.nodes())
    drug_nodes = [n for n in graph_nodes if G_nx.nodes[n].get('type') == 'drug']

    drug_scores = {}
    for drug_node in drug_nodes:
        idx = node_map_int[drug_node]
        drug_scores[drug_node] = float(
            F.cosine_similarity(z[idx].unsqueeze(0), disease_vec.unsqueeze(0)).item()
        )

    # Match trial drugs to graph nodes
    matched = []
    for _, row in trial_df.iterrows():
        trial_drug = row['drug']
        outcome = row['phase3_histology_outcome']
        kg_node = _fuzzy_match(trial_drug, drug_nodes)
        if kg_node and kg_node in drug_scores:
            matched.append({
                'trial_drug': trial_drug,
                'kg_node': kg_node,
                'gnn_score': drug_scores[kg_node],
                'outcome': outcome,
                'label': 1.0 if outcome == 'success' else (0.5 if outcome == 'mixed' else 0.0),
            })
        else:
            print(f"  WARNING: '{trial_drug}' not found in KG.")

    if len(matched) < 3:
        print(f"\nOnly {len(matched)} drugs matched. Cannot compute AUROC meaningfully.")
        return

    match_df = pd.DataFrame(matched)
    print(f"\nMatched {len(match_df)} drugs:")
    print(match_df[['trial_drug', 'kg_node', 'gnn_score', 'outcome']].to_string(index=False))

    # AUROC: success (1) vs failure (0), exclude mixed
    binary_df = match_df[match_df['outcome'].isin(['success', 'failure'])].copy()
    binary_df = binary_df.sort_values('gnn_score', ascending=False)

    if len(binary_df['outcome'].unique()) < 2:
        print("\nCannot compute AUROC: only one outcome class after filtering.")
        return

    from sklearn.metrics import roc_auc_score, roc_curve, average_precision_score

    y_true = (binary_df['outcome'] == 'success').astype(int).values
    y_scores = binary_df['gnn_score'].values

    auroc = roc_auc_score(y_true, y_scores)
    ap = average_precision_score(y_true, y_scores)

    # Bootstrap CI
    rng = np.random.default_rng(42)
    n_boot = 1000
    boot_aucs = []
    for _ in range(n_boot):
        idx = rng.choice(len(y_true), len(y_true), replace=True)
        if len(np.unique(y_true[idx])) < 2:
            continue
        boot_aucs.append(roc_auc_score(y_true[idx], y_scores[idx]))
    boot_aucs = np.array(boot_aucs)
    ci_lo, ci_hi = np.percentile(boot_aucs, [2.5, 97.5])

    print(f"\n{'='*60}")
    print(f"  TRIAL BENCHMARK RESULTS")
    print(f"{'='*60}")
    print(f"  AUROC: {auroc:.3f}  (95% CI: [{ci_lo:.3f}, {ci_hi:.3f}])")
    print(f"  AP:    {ap:.3f}")
    print(f"  n drugs: {len(binary_df)}")
    if auroc >= 0.70:
        print(f"  ✓ Strong result: framework separates winners from losers.")
    elif auroc >= 0.60:
        print(f"  ~ Moderate: some signal, interpret with caution.")
    else:
        print(f"  ⚠ Near-chance: embedding may not encode therapeutic efficacy.")
        print(f"    Consider reframing drug-screening as exploratory hypothesis generation.")
    print(f"{'='*60}")

    # ROC plot
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, 'o-', color='#457b9d', lw=2,
             label=f'GNN (AUROC={auroc:.3f})')
    plt.plot([0, 1], [0, 1], '--', color='gray')
    plt.fill_between(fpr, tpr, alpha=0.15, color='#457b9d')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Clinical Trial Benchmark: GNN Score vs Phase-3 Outcome')
    plt.legend()
    plt.tight_layout()
    roc_path = os.path.join(save_dir, 'trial_benchmark_roc.png')
    plt.savefig(roc_path, dpi=300)
    plt.close()
    print(f"Saved ROC → {roc_path}")

    # Save stats
    stats = {
        'auroc': float(auroc),
        'auroc_ci_lo': float(ci_lo),
        'auroc_ci_hi': float(ci_hi),
        'ap': float(ap),
        'n_drugs_matched': len(match_df),
        'n_binary': len(binary_df),
        'per_drug': match_df[['trial_drug', 'kg_node', 'gnn_score', 'outcome']].to_dict(orient='records'),
    }
    stats_path = os.path.join(save_dir, 'trial_benchmark_stats.json')
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"Saved stats → {stats_path}")


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='C2 Trial Benchmark')
    p.add_argument('--gexf', required=True)
    p.add_argument('--weights', required=True)
    p.add_argument('--save-dir', default='results')
    args = p.parse_args()
    run_benchmark(args.gexf, args.weights, args.save_dir)
