# -*- coding: utf-8 -*-
"""train.py — Training loop, evaluation, early stopping, seed sweep.

Canonical definitions (last variants from the original file):
  - train()  →  train_one_epoch()  (renamed to avoid shadowing)
  - evaluate()  (5-return-value version: acc, ap, auc, scores, labels)

Also adds:
  - Proper early stopping that actually *uses* the patience parameter
  - Multi-seed training wrapper
  - Bootstrap CI for AUROC/AUPRC
"""

import os
import re
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
    auc as sklearn_auc,
)

from masldgnn.model import GraphSAGE_LinkPredictor
from masldgnn.sampling import compute_pos_weight


def train_one_epoch(model, data, optimizer, device, pos_weight=None):
    """Single training step.  Returns loss float."""
    model.train()
    optimizer.zero_grad()
    z = model.encode(data.x, data.edge_index)
    out = model.decode(z, data.edge_label_index)
    target = data.edge_label.float().to(device)
    loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)(out, target)
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def evaluate(model, data, device,
              test_edge_label_index=None, test_edge_label=None):
    """Evaluate model.  Returns (acc, ap, roc_auc, scores_np, labels_np)."""
    model.eval()
    z = model.encode(data.x, data.edge_index)
    eval_idx = test_edge_label_index if test_edge_label_index is not None else data.edge_label_index
    eval_lbl = test_edge_label if test_edge_label is not None else data.edge_label

    if eval_idx.size(1) == 0:
        return 0.5, 0.5, 0.5, np.array([]), np.array([])

    out = model.decode(z, eval_idx)
    scores = torch.sigmoid(out)
    scores_np = scores.cpu().numpy()
    target_np = eval_lbl.cpu().numpy()

    if len(np.unique(target_np)) < 2:
        return 0.5, 0.5, 0.5, scores_np, target_np

    ap = average_precision_score(target_np, scores_np)
    roc_auc = roc_auc_score(target_np, scores_np)
    acc = ((scores_np > 0.5).astype(int) == target_np).mean()
    return acc, ap, roc_auc, scores_np, target_np


def train_model(data, val_data, in_dim, gnn_dims, device,
                num_epochs=200, patience=20, lr=0.005, weight_decay=5e-4,
                save_dir="results"):
    """Full training loop with real early stopping.

    Returns
    -------
    model : trained GraphSAGE_LinkPredictor
    history : dict of epoch-indexed lists
    """
    os.makedirs(save_dir, exist_ok=True)

    if len(gnn_dims) == 3:
        model = GraphSAGE_LinkPredictor(in_dim, gnn_dims[0], gnn_dims[1],
                                          gnn_dims[2]).to(device)
    else:
        model = GraphSAGE_LinkPredictor(in_dim, gnn_dims[0], gnn_dims[1]).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    pw = compute_pos_weight(data.edge_label).to(device)

    best_val_auc, best_epoch, counter = 0.0, 0, 0
    best_state = None
    history = defaultdict(list)

    for epoch in range(1, num_epochs + 1):
        loss = train_one_epoch(model, data, optimizer, device, pw)
        _, val_ap, val_auc, _, _ = evaluate(model, val_data, device)

        history["epoch"].append(epoch)
        history["loss"].append(loss)
        history["val_ap"].append(val_ap)
        history["val_auc"].append(val_auc)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch = epoch
            counter = 0
            best_state = model.state_dict()
        else:
            counter += 1

        if epoch % 20 == 0:
            print(f"    Epoch {epoch:03d} | Loss: {loss:.4f} | "
                  f"Val AUC: {val_auc:.3f} | Val AP: {val_ap:.3f}")

        if counter >= patience:
            print(f"  Early stopping at epoch {epoch} "
                  f"(best {best_epoch}, Val AUC={best_val_auc:.3f})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, dict(history)


def seed_sweep(train_data, val_data, test_data, in_dim, gnn_dims, device,
               seeds, save_dir="results", **train_kwargs):
    """Train across multiple seeds; return per-seed + aggregated metrics."""
    all_metrics = []
    for seed in seeds:
        from masldgnn.config import set_seed
        set_seed(seed)
        print(f"\n--- Seed {seed} ---")
        model, history = train_model(
            train_data, val_data, in_dim, gnn_dims, device,
            save_dir=os.path.join(save_dir, f"seed_{seed}"),
            **train_kwargs,
        )
        acc, ap, auc, scores, labels = evaluate(model, test_data, device)
        print(f"  Seed {seed}: Acc={acc:.3f}  AP={ap:.3f}  AUC={auc:.3f}")
        all_metrics.append({"seed": seed, "acc": acc, "ap": ap, "auc": auc})

    import pandas as pd
    df = pd.DataFrame(all_metrics)
    print(f"\n=== Seed Sweep Summary ===")
    print(df.round(3).to_string(index=False))
    print(f"  Mean AUC: {df['auc'].mean():.3f} +/- {df['auc'].std():.3f}")
    return df


def bootstrap_ci(scores_np, labels_np, n_boot=1000, alpha=0.05):
    """Bootstrap 95% CI for AUROC."""
    rng = np.random.default_rng(42)
    aucs = []
    n = len(labels_np)
    for _ in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        s, l = scores_np[idx], labels_np[idx]
        if len(np.unique(l)) < 2:
            continue
        aucs.append(roc_auc_score(l, s))
    aucs = np.array(aucs)
    lo, hi = np.percentile(aucs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def save_training_plots(history, title, save_dir):
    """Save training-loss + validation-metric plots."""
    os.makedirs(save_dir, exist_ok=True)
    title_safe = re.sub(r'[\\/*?:"<>|]', "", title)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history["epoch"], history["loss"], label="Training Loss")
    plt.title(f"Loss ({title_safe})")
    plt.xlabel("Epoch"); plt.ylabel("BCE Loss"); plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history["epoch"], history["val_auc"], label="Val AUC")
    plt.plot(history["epoch"], history["val_ap"], label="Val AP")
    plt.title(f"Validation Metrics ({title_safe})")
    plt.xlabel("Epoch"); plt.ylabel("Score"); plt.legend()

    plt.tight_layout()
    path = os.path.join(save_dir, f"{title_safe}_training_curves.png")
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"Saved training curves → {path}")


def save_performance_plots(y_true, y_scores, title, save_dir):
    """Save ROC + PR curves."""
    os.makedirs(save_dir, exist_ok=True)
    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        print(f"Skipping performance plots: not enough labels.")
        return
    title_safe = re.sub(r'[\\/*?:"<>|]', "", title)
    plt.figure(figsize=(12, 5))

    try:
        plt.subplot(1, 2, 1)
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_val = sklearn_auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f"ROC (AUC={roc_val:.3f})")
        plt.plot([0, 1], [0, 1], "--", color="navy", lw=1)
        plt.xlabel("FPR"); plt.ylabel("TPR")
        plt.title(f"ROC ({title_safe})"); plt.legend(loc="lower right")
    except Exception as e:
        print(f"ROC plot error: {e}")

    try:
        plt.subplot(1, 2, 2)
        prec, rec, _ = precision_recall_curve(y_true, y_scores)
        ap = average_precision_score(y_true, y_scores)
        plt.plot(rec, prec, lw=2, label=f"PR (AP={ap:.3f})")
        plt.xlabel("Recall"); plt.ylabel("Precision")
        plt.title(f"PR ({title_safe})"); plt.legend(loc="upper right")
    except Exception as e:
        print(f"PR plot error: {e}")

    plt.tight_layout()
    path = os.path.join(save_dir, f"{title_safe}_performance_curves.png")
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"Saved performance curves → {path}")
