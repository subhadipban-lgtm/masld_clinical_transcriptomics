# -*- coding: utf-8 -*-
from __future__ import annotations
"""diagnostics.py — C7 embedding-collapse checks + calibration.

Can be called from 15_embedding_diagnostics.py or from 09c_evaluate.py.
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F


def embedding_diagnostics(drug_z: torch.Tensor, G_nx: nx.Graph,
                             drug_indices: list[int],
                             int_to_label: dict, save_dir: str = "results"):
    """Run all C7 checks on a drug embedding tensor.

    Returns dict of stats.
    """
    os.makedirs(save_dir, exist_ok=True)
    drug_labels = [int_to_label.get(i, f"node_{i}") for i in drug_indices]

    # Pairwise cosine
    Zn = F.normalize(drug_z, dim=1)
    S = (Zn @ Zn.T).numpy()
    off = S[~np.eye(len(S), dtype=bool)]

    # Effective rank
    sv = torch.linalg.svdvals(drug_z - drug_z.mean(0))
    pr = (sv.sum() ** 2) / (sv ** 2).sum()

    print(f"\n{'='*60}")
    print(f"  EMBEDDING DIAGNOSTICS  ({len(drug_labels)} drug nodes)")
    print(f"{'='*60}")
    print(f"  mean cosine:  {off.mean():.4f}")
    print(f"  median cosine: {np.median(off):.4f}")
    print(f"  frac > 0.99:  {(off > 0.99).mean():.4f}")
    print(f"  norms min/med/max: "
          f"{drug_z.norm(dim=1).min().item():.3f} / "
          f"{drug_z.norm(dim=1).median().item():.3f} / "
          f"{drug_z.norm(dim=1).max().item():.3f}")
    print(f"  effective rank (of {drug_z.size(1)}): {pr.item():.2f}")

    # Per-drug degree
    for d in ["Resmetirom", "Liraglutide", "Semaglutide",
              "Obeticholic acid", "Disulfiram", "Sulfinpyrazone",
              "Balsalazide", "Olsalazine"]:
        matches = [n for n in G_nx.nodes() if d.lower() in n.lower()]
        if matches:
            print(f"    {matches[0]:40s}  degree = {G_nx.degree(matches[0])}")
        else:
            print(f"    {d:40s}  ABSENT")

    collapsed = off.mean() > 0.9 or pr.item() < 3
    if collapsed:
        print("  ⚠  WARNING: Embedding space appears COLLAPSED.")
    else:
        print("  ✓  Embedding space appears differentiated.")

    # Histogram
    plt.figure(figsize=(8, 5))
    plt.hist(off, bins=50, color="#457b9d", edgecolor="white")
    plt.xlabel("Pairwise Cosine Similarity")
    plt.ylabel("Count")
    plt.title("Drug Embedding Pairwise Cosine Similarity")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "drug_similarity_distribution.png"), dpi=300)
    plt.close()

    stats = {
        "n_drugs": len(drug_labels),
        "mean_cosine": float(off.mean()),
        "median_cosine": float(np.median(off)),
        "frac_gt_099": float((off > 0.99).mean()),
        "norm_min": float(drug_z.norm(dim=1).min().item()),
        "norm_median": float(drug_z.norm(dim=1).median().item()),
        "norm_max": float(drug_z.norm(dim=1).max().item()),
        "effective_rank": float(pr.item()),
        "embedding_dim": int(drug_z.size(1)),
        "collapsed": bool(collapsed),
    }
    with open(os.path.join(save_dir, "embedding_diagnostics.json"), "w") as f:
        json.dump(stats, f, indent=2)

    return stats


def calibration_curve(scores_np, labels_np, n_bins=10,
                        save_dir="results", title=""):
    """Reliability diagram + Brier score."""
    os.makedirs(save_dir, exist_ok=True)
    bins = np.linspace(0, 1, n_bins + 1)
    brier = np.mean((scores_np - labels_np) ** 2)

    bin_accs, bin_confs, bin_sizes = [], [], []
    for i in range(n_bins):
        mask = (scores_np >= bins[i]) & (scores_np < bins[i + 1])
        if i == n_bins - 1:
            mask = (scores_np >= bins[i]) & (scores_np <= bins[i + 1])
        if mask.sum() == 0:
            continue
        bin_confs.append(scores_np[mask].mean())
        bin_accs.append(labels_np[mask].mean())
        bin_sizes.append(mask.sum())

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], '--', color='gray', label='Perfect calibration')
    ax.plot(bin_confs, bin_accs, 'o-', color='#457b9d', lw=2)
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.set_title(f"Reliability Diagram (Brier={brier:.4f}) {title}")
    ax.legend()
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    plt.tight_layout()
    path = os.path.join(save_dir, "calibration_curve.png")
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"Saved calibration curve → {path}  Brier={brier:.4f}")
    return {"brier_score": float(brier)}
