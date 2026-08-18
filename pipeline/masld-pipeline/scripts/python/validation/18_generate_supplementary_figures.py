#!/usr/bin/env python3
"""
18_generate_supplementary_figures.py

Generates Supplementary Figures 15-18 for the MASLD manuscript.

Usage:
    python 18_generate_supplementary_figures.py \
        --longitudinal data/supplementary/longitudinal_paired.csv \
        --proteomics data/supplementary/proteomics_paired.csv \
        --output figures/supplementary/
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.5)


def load_longitudinal(path):
    """Load longitudinal paired data."""
    if not Path(path).exists():
        raise FileNotFoundError(f"Longitudinal data file not found: {path}")
    df = pd.read_csv(path)
    required = ['patient_id', 'trajectory', 'delta_axis', 'delta_nas', 'delta_fibrosis']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Longitudinal data missing required columns: {missing}")
    return df


def load_proteomics(path):
    """Load proteomics paired data."""
    if not Path(path).exists():
        raise FileNotFoundError(f"Proteomics data file not found: {path}")
    df = pd.read_csv(path)
    required = ['patient_id', 'fibrosis_stage', 'transcript_score', 'protein_score']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Proteomics data missing required columns: {missing}")
    return df


def figure15(long_df, out_dir):
    """Generate Figure 15: Longitudinal dynamics."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    order = ['Progression', 'Stable', 'Regression']
    palette = {'Progression': '#d9534f', 'Stable': '#6c757d', 'Regression': '#5cb85c'}
    sns.boxplot(
        data=long_df, x='trajectory', y='delta_axis', order=order, ax=axes[0],
        hue='trajectory', palette=palette, legend=False
    )
    sns.stripplot(
        data=long_df, x='trajectory', y='delta_axis', order=order, ax=axes[0],
        color='black', alpha=0.5, size=6
    )
    axes[0].axhline(0, linestyle='--', color='black', alpha=0.3)
    axes[0].set_title('(a) Ferroptosis Tracks Disease Dynamics')
    axes[0].set_ylabel('Δ Ferroptosis Axis Score (Follow-up - Baseline)')
    axes[0].set_xlabel('')

    prog = long_df[long_df['trajectory'] == 'Progression']['delta_axis']
    reg = long_df[long_df['trajectory'] == 'Regression']['delta_axis']
    _, p_val = stats.mannwhitneyu(prog, reg, alternative='two-sided')
    axes[0].text(0.5, 0.95, f'p = {p_val:.4f}', transform=axes[0].transAxes, ha='center', fontweight='bold')

    sns.regplot(
        data=long_df, x='delta_nas', y='delta_axis', ax=axes[1],
        scatter_kws={'alpha': 0.6}, line_kws={'color': 'red'}
    )
    r_nas, p_nas = stats.pearsonr(long_df['delta_nas'], long_df['delta_axis'])
    axes[1].set_title(f'(b) Correlation with NAS Change (r = {r_nas:.2f}, p = {p_nas:.2e})')
    axes[1].set_xlabel('Δ NAS (Follow-up - Baseline)')
    axes[1].set_ylabel('Δ Ferroptosis Axis Score')

    sns.regplot(
        data=long_df, x='delta_fibrosis', y='delta_axis', ax=axes[2],
        scatter_kws={'alpha': 0.6}, line_kws={'color': 'red'}
    )
    r_fib, p_fib = stats.pearsonr(long_df['delta_fibrosis'], long_df['delta_axis'])
    axes[2].set_title(f'(c) Correlation with ΔFibrosis Stage (r = {r_fib:.2f}, p = {p_fib:.2e})')
    axes[2].set_xlabel('Δ Fibrosis Stage (Follow-up - Baseline)')
    axes[2].set_ylabel('Δ Ferroptosis Axis Score')

    plt.tight_layout()
    out_path = out_dir / 'Figure_15_Ferroptosis_Longitudinal_Dynamics.pdf'
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved Figure 15 to {out_path}")


def figure16(long_df, out_dir):
    """Generate Figure 16: Individual trajectories."""
    fig, ax = plt.subplots(figsize=(8, 6))

    colors = {'Progression': '#d9534f', 'Regression': '#5cb85c', 'Stable': '#6c757d'}
    for _, row in long_df.iterrows():
        ax.arrow(
            row['baseline_nas'], row['baseline_axis'],
            row['followup_nas'] - row['baseline_nas'],
            row['followup_axis'] - row['baseline_axis'],
            head_width=0.05, head_length=0.05,
            fc=colors[row['trajectory']], ec=colors[row['trajectory']],
            alpha=0.7, length_includes_head=True
        )
    for _, row in long_df.iterrows():
        ax.scatter(
            row['baseline_nas'], row['baseline_axis'],
            s=40, facecolors='none', edgecolors=colors[row['trajectory']]
        )
        ax.scatter(
            row['followup_nas'], row['followup_axis'],
            s=40, color=colors[row['trajectory']]
        )

    from matplotlib.patches import Patch
    n_prog = len(long_df[long_df['trajectory'] == 'Progression'])
    n_reg = len(long_df[long_df['trajectory'] == 'Regression'])
    n_stab = len(long_df[long_df['trajectory'] == 'Stable'])
    legend_elements = [
        Patch(facecolor='#d9534f', label=f'Progression (n={n_prog})'),
        Patch(facecolor='#5cb85c', label=f'Regression (n={n_reg})'),
        Patch(facecolor='#6c757d', label=f'Stable (n={n_stab})')
    ]
    ax.legend(handles=legend_elements, loc='lower right')

    ax.set_xlabel('NAS Score')
    ax.set_ylabel('Ferroptosis Axis Score')
    ax.set_title('Individual Patient Trajectories')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path = out_dir / 'Figure_16_Individual_Trajectories.pdf'
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved Figure 16 to {out_path}")


def figure17(proteomics_df, out_dir):
    """Generate Figure 17: Transcriptomics-proteomics correlation and ROC."""
    t_cols = [c for c in proteomics_df.columns if c.endswith('_t')]
    p_cols = [c for c in proteomics_df.columns if c.endswith('_p')]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    if t_cols and p_cols:
        corrs = []
        for t, p in zip(sorted(t_cols), sorted(p_cols)):
            gene = t[:-2]
            if p == gene + '_p':
                corr, _ = stats.pearsonr(proteomics_df[t], proteomics_df[p])
                corrs.append(corr)
        if corrs:
            axes[0].hist(corrs, bins=20, color='skyblue', edgecolor='black')
            axes[0].axvline(np.median(corrs), color='red', linestyle='--', label=f'Median = {np.median(corrs):.2f}')
            axes[0].set_xlabel('Transcript-Protein Correlation (r)')
            axes[0].set_ylabel('Frequency')
            axes[0].set_title('(a) 57-Gene Signature: Transcript vs Protein')
            axes[0].legend()
            pos_frac = np.mean(np.array(corrs) > 0)
            axes[0].text(0.7, 0.9, f'Positive: {pos_frac:.0%}', transform=axes[0].transAxes)
    else:
        axes[0].text(0.5, 0.5, 'Gene-level data not available', ha='center', va='center')
        axes[0].set_title('(a) Transcript-Protein Correlation')

    y_true = (proteomics_df['fibrosis_stage'] >= 3).astype(int)
    for label, score in [('Transcriptomics', 'transcript_score'), ('Proteomics', 'protein_score')]:
        if score in proteomics_df.columns:
            fpr, tpr, _ = roc_curve(y_true, proteomics_df[score])
            roc_auc = auc(fpr, tpr)
            axes[1].plot(fpr, tpr, label=f'{label} (AUC = {roc_auc:.2f})')
    axes[1].plot([0, 1], [0, 1], 'k--', alpha=0.3)
    axes[1].set_xlabel('False Positive Rate')
    axes[1].set_ylabel('True Positive Rate')
    axes[1].set_title('(b) Proteomics Prediction of Advanced Fibrosis (F3-F4)')
    axes[1].legend(loc='lower right')

    plt.tight_layout()
    out_path = out_dir / 'Figure_17_Transcriptomics_Proteomics_Correlation.pdf'
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved Figure 17 to {out_path}")


def figure18(proteomics_df, out_dir):
    """Generate Figure 18: Cross-omics correlation and Bland-Altman."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    x = proteomics_df['transcript_score']
    y = proteomics_df['protein_score']
    r, p = stats.pearsonr(x, y)
    colors = np.where(proteomics_df['fibrosis_stage'] >= 3, '#d9534f', '#0275d8')
    axes[0].scatter(x, y, c=colors, alpha=0.6, edgecolors='k', linewidth=0.5)
    slope, intercept, _, _, _ = stats.linregress(x, y)
    x_range = np.linspace(x.min(), x.max(), 100)
    axes[0].plot(x_range, slope * x_range + intercept, 'k--', label=f'r = {r:.2f}, p = {p:.2e}')
    axes[0].set_xlabel('Transcriptomics Ferroptosis Score')
    axes[0].set_ylabel('Proteomics Ferroptosis Score')
    axes[0].set_title('(a) Cross-Omics Correlation (n = 120 paired samples)')
    from matplotlib.patches import Patch
    axes[0].legend(handles=[Patch(color='#0275d8', label='F0-F2'), Patch(color='#d9534f', label='F3-F4')])

    mean_val = (x + y) / 2
    diff = y - x
    mean_diff = np.mean(diff)
    std_diff = np.std(diff)
    axes[1].scatter(mean_val, diff, alpha=0.6, edgecolors='k', linewidth=0.5)
    axes[1].axhline(mean_diff, color='black', linestyle='-')
    axes[1].axhline(mean_diff + 1.96 * std_diff, color='red', linestyle='--', label='+1.96 SD')
    axes[1].axhline(mean_diff - 1.96 * std_diff, color='red', linestyle='--', label='-1.96 SD')
    axes[1].set_xlabel('Mean of Transcript & Protein Scores')
    axes[1].set_ylabel('Difference (Protein - Transcript)')
    axes[1].set_title('(b) Bland-Altman Agreement')
    axes[1].legend()

    plt.tight_layout()
    out_path = out_dir / 'Figure_18_CrossOmics_Validation.pdf'
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved Figure 18 to {out_path}")


def main():
    parser = argparse.ArgumentParser(description='Generate supplementary figures 15-18')
    parser.add_argument('--longitudinal', required=True, help='CSV file with longitudinal data')
    parser.add_argument('--proteomics', required=True, help='CSV file with paired transcript/protein data')
    parser.add_argument('--output', default='figures/', help='Output directory for figures')
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    long_df = load_longitudinal(args.longitudinal)
    prot_df = load_proteomics(args.proteomics)

    figure15(long_df, out_dir)
    figure16(long_df, out_dir)
    figure17(prot_df, out_dir)
    figure18(prot_df, out_dir)

    print("\nAll supplementary figures generated successfully.")


if __name__ == '__main__':
    main()
