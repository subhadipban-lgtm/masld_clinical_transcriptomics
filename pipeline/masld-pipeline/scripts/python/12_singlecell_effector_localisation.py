#!/usr/bin/env python3
"""
WS28 analysis 2.10 — ferroptosis effector-gene localisation in the GSE136103 atlas
(Ramachandran et al. 2019; 10 human liver donors, FACS fractions; blood/mouse excluded).

Pre-committed rules (printed first):
  R1: cell types assigned per cell by argmax of the mean within-sample-set z-score of a
      fixed canonical marker panel (panel below); liver fractions only.
  R2: per DONOR x cell type, pseudobulk mean of log1p(CPM) for each effector gene
      (cells are never replicates, AGENTS.md section 6). HSC vs hepatocyte per gene =
      Wilcoxon signed-rank across donors with both types present; BH across genes.
  R3: report per-gene HSC/hepatocyte log2 ratio of donor pseudobulk means; the cell-level
      heatmap is explicitly descriptive only.
Effector panel: GPX4, ACSL4, SLC7A11, SLC3A2, TFRC, NFE2L2, SAT1, LPCAT3, ALOX15, PRNP,
DHODH, CISD1, SLC7A5, STEAP3.
Output -> results/ws28_sc_effectors/
"""
import gzip
import json
import os
import re
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy.io import mmread
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

SRC = "data/singlecell/GSE136103"
OUT = "results/ws28_sc_effectors"
os.makedirs(OUT, exist_ok=True)

MARKERS = {
    "hepatocyte": ["ALB", "TTR", "CYP2E1", "PCK1", "AGXT"],
    "HSC_mesenchymal": ["DCN", "PDGFRB", "COL1A1", "COL3A1", "LHX9", "RGS5"],
    "LSEC": ["PECAM1", "VWF", "PLVAP", "ACKR1"],
    "cholangiocyte": ["KRT7", "KRT19", "EPCAM", "SOX9"],
    "macrophage": ["CD68", "C1QA", "C1QB", "LYZ", "MRC1"],
    "T_NK": ["CD3D", "TRAC", "NKG7", "GNLY"],
    "B_cell": ["MS4A1", "CD79A"],
    "erythroid": ["HBB", "HBA1", "ALAS2"],
}
EFFECTORS = ["GPX4", "ACSL4", "SLC7A11", "SLC3A2", "TFRC", "NFE2L2", "SAT1", "LPCAT3",
             "ALOX15", "PRNP", "DHODH", "CISD1", "SLC7A5", "STEAP3"]
KEEP = sorted({g for v in MARKERS.values() for g in v} | set(EFFECTORS))

print("R1: argmax mean-z marker assignment, liver fractions only (blood/mouse excluded).")
print("R2: donor x cell-type pseudobulk log1p(CPM); HSC vs hepatocyte = Wilcoxon signed-rank")
print("    across donors; BH across genes; cells never treated as replicates.")
print("R3: log2 HSC/hep ratios reported; cell-level heatmap descriptive only.")

samples = sorted({f[:-len("_genes.tsv.gz")] for f in os.listdir(SRC) if f.endswith("_genes.tsv.gz")})
liver = [s for s in samples
         if re.search(r"(healthy|cirrhotic)\d", s) and "blood" not in s and "mouse" not in s]
assert len(liver) == 20, f"expected 20 liver fraction samples (11 healthy + 9 cirrhotic), got {len(liver)}"
print(f"\nliver fractions: {len(liver)} across donors:",
      sorted({re.search(r'(healthy|cirrhotic)\d', s).group(0) for s in liver}))

cell_rows = []   # per-cell effector log1p CPM + assigned type + donor
manifest = []
for s in liver:
    donor = re.search(r"(healthy|cirrhotic)\d", s).group(0)
    genes = [l.strip().split("\t")[1].upper() for l in gzip.open(f"{SRC}/{s}_genes.tsv.gz", "rt")]
    M = mmread(f"{SRC}/{s}_matrix.mtx.gz").tocsc()          # genes x cells
    lib = np.asarray(M.sum(axis=0)).ravel()
    keep_lib = lib > 500
    cpm = M.multiply(1e6 / np.maximum(lib, 1)).tocsc()
    idx = {g: i for i, g in enumerate(genes)}
    sub = cpm[[idx[g] for g in KEEP if g in idx]]
    have = [g for g in KEEP if g in idx]
    E = np.log1p(np.asarray(sub.todense()))                  # genes_kept x cells
    E = E[:, keep_lib]
    Ez = (E - E.mean(axis=1, keepdims=True)) / np.where(E.std(axis=1, keepdims=True) > 0,
                                                       E.std(axis=1, keepdims=True), 1)
    gd = {g: k for k, g in enumerate(have)}
    scores = {}
    for t, ms in MARKERS.items():
        ms = [m for m in ms if m in gd]
        if len(ms) >= 2:
            scores[t] = Ez[[gd[m] for m in ms]].mean(axis=0)
    S = pd.DataFrame(scores)
    celltype = S.idxmax(axis=1).to_numpy()
    n_cells = int(keep_lib.sum())
    manifest.append({"sample": s, "donor": donor, "cells": n_cells})
    for j, g in enumerate(EFFECTORS):
        if g in gd:
            cell_rows.append(pd.DataFrame({"donor": donor, "celltype": celltype,
                                           "gene": g,
                                           "expr": E[gd[g]]}))
    print(f"  {s:28s} {n_cells:6d} cells | types: "
          f"{pd.Series(celltype).value_counts().to_dict()}")

cells = pd.concat(cell_rows, ignore_index=True)
cells.to_csv(f"{OUT}/cell_level_effector_expression.csv", index=False)

# donor x celltype pseudobulk
pb = cells.groupby(["donor", "celltype", "gene"]).expr.mean().reset_index()
pb.to_csv(f"{OUT}/donor_pseudobulk_effectors.csv", index=False)
print("\ndonor x celltype pseudobulk rows:", len(pb),
      "| donors:", pb.donor.nunique(), "| types:", sorted(pb.celltype.unique()))

# HSC vs hepatocyte (R2)
res = []
for g in EFFECTORS:
    w = pb[pb.gene == g].pivot_table(index="donor", columns="celltype", values="expr")
    if not {"HSC_mesenchymal", "hepatocyte"}.issubset(w.columns):
        continue
    w = w.dropna(subset=["HSC_mesenchymal", "hepatocyte"])
    if len(w) < 4:
        res.append({"gene": g, "n_donors": len(w), "note": "too few paired donors"})
        continue
    stat, p = wilcoxon(w.HSC_mesenchymal, w.hepatocyte)
    res.append({"gene": g, "n_donors": len(w),
                "HSC_mean": w.HSC_mesenchymal.mean(), "hep_mean": w.hepatocyte.mean(),
                "log2_HSC_over_hep": np.log2((w.HSC_mesenchymal.mean() + 1e-9) /
                                             (w.hepatocyte.mean() + 1e-9)),
                "wilcoxon_p": p})
R = pd.DataFrame(res)
tested = R.dropna(subset=["wilcoxon_p"])
R.loc[tested.index, "p_adj"] = multipletests(tested.wilcoxon_p, method="fdr_bh")[1]
R = R.sort_values("log2_HSC_over_hep", ascending=False)
R.to_csv(f"{OUT}/effector_HSC_vs_hepatocyte.csv", index=False)
print(R.round(4).to_string(index=False))

# descriptive cell-level heatmap (R3 label)
hm = cells.pivot_table(index="gene", columns="celltype", values="expr", aggfunc="mean")
hm = hm.reindex([g for g in EFFECTORS if g in hm.index])
fig, ax = plt.subplots(figsize=(9, 7))
sns.heatmap(hm, cmap="viridis", ax=ax, annot=True, fmt=".2f", cbar_kws={"label": "mean log1p CPM"})
ax.set_title("Ferroptosis effector genes by cell type — GSE136103\n"
             "DESCRIPTIVE cell-level means (inferential tests use donor pseudobulk, R2)")
plt.tight_layout()
fig.savefig(f"{OUT}/fig_effector_celltype_heatmap.png", dpi=300)
plt.close(fig)


def md5(p):
    import hashlib
    h = hashlib.md5()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


prov = {"output": OUT + "/", 
        "derived_from": [{"path": "data/singlecell/GSE136103_RAW.tar",
                          "md5": md5("data/singlecell/GSE136103_RAW.tar"),
                          "bytes": os.path.getsize("data/singlecell/GSE136103_RAW.tar"),
                          "url": "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE136nnn/GSE136103/suppl/GSE136103_RAW.tar"}],
        "script": "scripts/python/46_ws28_sc_effectors.py",
        "git_commit": os.popen("git rev-parse --short HEAD").read().strip(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "rules": "R1 argmax marker; R2 donor pseudobulk + Wilcoxon signed-rank, BH across genes; R3 log2 ratios",
        "marker_panel": MARKERS, "effector_panel": EFFECTORS,
        "samples": manifest}
with open(f"{OUT}/.extraction.provenance.json", "w") as f:
    json.dump(prov, f, indent=2)
pd.DataFrame(manifest).to_csv(f"{OUT}/load_manifest.csv", index=False)

stats = {"rules": "R1-R3 above",
         "donors": int(pb.donor.nunique()),
         "significant_HSC_vs_hep_FDR05": R[R.p_adj < 0.05].gene.tolist(),
         "results": {r.gene: {"log2_HSC_over_hep": getattr(r, "log2_HSC_over_hep", None),
                              "p": getattr(r, "wilcoxon_p", None),
                              "p_adj": getattr(r, "p_adj", None),
                              "n_donors": r.n_donors}
                     for r in R.itertuples()},
         "analysis_date": datetime.now(timezone.utc).isoformat()}
with open(f"{OUT}/stats_ws28_sc_effectors.json", "w") as f:
    json.dump(stats, f, indent=2, default=float)
print("\nDONE — results/ws28_sc_effectors/")
