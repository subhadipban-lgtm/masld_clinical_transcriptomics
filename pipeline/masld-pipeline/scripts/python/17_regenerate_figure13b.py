# -*- coding: utf-8 -*-
"""17_regenerate_figure13b.py — Honest Figure 13B regeneration.

Reads Pioglitazone_hydrochloride_PPARG_subgraph_nodes.csv and plots
the actual top-10 nodes by GNNExplainer Importance score.

AGENT.md Rule 2: Never label an inferred value as model output.
                 If a figure needs an entity the model did not score, OMIT it.
AGENT.md Rule 1: Never hardcode or estimate any scientific value.

Explicitly EXCLUDED (not in CSV):
  - RXRA
  - "PPAR Signaling" (pathway node)
  - "Estimated (biologically inferred)" category

Usage:
    python scripts/python/17_regenerate_figure13b.py \
        --csv data/figure_sources/Pioglitazone_hydrochloride_PPARG_subgraph_nodes.csv \
        --out figures/Figure_13B_revised.pdf
"""

import argparse, json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Forbidden entities — MUST NOT appear in figure (not in GNNExplainer output)
# ---------------------------------------------------------------------------
FORBIDDEN_NODES = {"RXRA", "PPAR Signaling", "PPAR signaling"}

TYPE_COLORS = {
    "drug":      "#e63946",
    "gene":      "#457b9d",
    "mechanism": "#2a9d8f",
}
TYPE_LABELS = {
    "drug":      "Drug node",
    "gene":      "Gene node",
    "mechanism": "Mechanism node",
}


def load_csv(csv_path):
    df = pd.read_csv(csv_path)
    required = {"NodeID", "Type", "Importance"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")
    df = df.dropna(subset=["NodeID", "Importance"])
    return df


def validate_no_forbidden(df):
    """Crash if any forbidden entity slipped in (AGENT.md Rule 2)."""
    in_data = set(df["NodeID"].astype(str))
    overlap = FORBIDDEN_NODES & in_data
    if overlap:
        raise ValueError(
            f"FORBIDDEN entities present in CSV data: {overlap}\n"
            "This violates AGENT.md Rule 2. Do not proceed."
        )


def get_top_n(df, n=10):
    return df.nlargest(n, "Importance").reset_index(drop=True)


def make_figure13b(df, out_path, csv_path, top_n=10):
    validate_no_forbidden(df)

    top = get_top_n(df, top_n)

    print(f"\n  Top {top_n} nodes by GNNExplainer Importance:")
    print(f"  {'NodeID':40s}  {'Type':12s}  Importance")
    print(f"  {'-'*40}  {'-'*12}  ----------")
    for _, row in top.iterrows():
        print(f"  {str(row['NodeID']):40s}  {str(row['Type']):12s}  {row['Importance']:.7f}")

    # Confirm PPARG value matches CSV (AGENT.md Rule 3)
    pparg_rows = top[top["NodeID"].str.upper() == "PPARG"]
    if not pparg_rows.empty:
        pparg_val = pparg_rows.iloc[0]["Importance"]
        print(f"\n  PPARG Importance (exact from CSV): {pparg_val:.7f}")
        print(f"  (Manuscript claimed 2.3431 — CSV has {pparg_val:.4f}; CSV wins per AGENT.md Rule 3)")

    # Build horizontal bar chart (descending order = top at top)
    top_sorted = top.sort_values("Importance", ascending=True)  # ascending for barh
    node_labels = top_sorted["NodeID"].astype(str).tolist()
    importances = top_sorted["Importance"].values
    types = top_sorted["Type"].astype(str).str.lower().tolist()
    colors = [TYPE_COLORS.get(t, "#888888") for t in types]

    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.barh(range(len(node_labels)), importances,
                   color=colors, edgecolor="white", height=0.7)

    # Value labels
    for bar, val in zip(bars, importances):
        ax.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", ha="left", fontsize=8)

    ax.set_yticks(range(len(node_labels)))
    ax.set_yticklabels(node_labels, fontsize=9)
    ax.set_xlabel("GNNExplainer Node Importance Score", fontsize=10)
    ax.set_title(
        f"Figure 13B — Top {top_n} Nodes in Pioglitazone-PPARG Subgraph\n"
        f"(GNNExplainer; all values from source CSV)\n"
        "RXRA and 'PPAR Signaling' excluded — not in GNNExplainer output",
        fontsize=10, fontweight="bold"
    )

    # Legend
    legend_patches = [
        mpatches.Patch(color=TYPE_COLORS[t], label=TYPE_LABELS[t])
        for t in sorted(TYPE_COLORS)
        if t in set(types)
    ]
    ax.legend(handles=legend_patches, loc="lower right", fontsize=8)

    # Source note
    ax.text(
        0.01, -0.13,
        f"Source: {os.path.basename(csv_path)}  |  "
        "No estimated/inferred values used  |  "
        f"Total nodes in subgraph: {len(df)}",
        transform=ax.transAxes, fontsize=7, color="#555555",
        ha="left"
    )

    ax.set_xlim(0, importances.max() * 1.18)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()

    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved Figure 13B -> {out_path}")

    # Stats JSON
    stats_dict = {
        "source_csv": os.path.abspath(csv_path),
        "total_nodes_in_subgraph": int(len(df)),
        "top_n": top_n,
        "excluded_entities": sorted(FORBIDDEN_NODES),
        "top_nodes": [
            {"node_id": str(r["NodeID"]),
             "type": str(r["Type"]),
             "importance": float(r["Importance"])}
            for _, r in top.iterrows()
        ]
    }
    stats_path = os.path.join(out_dir if out_dir else ".", "stats_figure13b.json")
    with open(stats_path, "w") as f:
        json.dump(stats_dict, f, indent=2)
    print(f"Saved stats    -> {stats_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True,
                        help="Pioglitazone_hydrochloride_PPARG_subgraph_nodes.csv")
    parser.add_argument("--out", default="figures/Figure_13B_revised.pdf")
    parser.add_argument("--top-n", type=int, default=10)
    args = parser.parse_args()

    print(f"Loading CSV: {args.csv}")
    df = load_csv(args.csv)
    print(f"  {len(df)} rows loaded.")
    make_figure13b(df, args.out, args.csv, args.top_n)


if __name__ == "__main__":
    main()
