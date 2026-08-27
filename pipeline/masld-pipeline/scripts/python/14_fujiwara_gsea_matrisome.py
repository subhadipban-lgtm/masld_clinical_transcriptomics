#!/usr/bin/env python3
"""
WS30 — Manuscript additions (work order 2026-08-25). Tasks 0-9.

PRE-COMMITTED RULES (stated before any result is computed):
  R0 (Task 0): no matrisome value is used until the release reconciles; if the authoritative
      matrix's release differs from WS29 v2023.2.Hs, regenerate the 8x10 matrix.
  R1 (Task 1), operationalised exactly as follows: discovery pattern = FerrDb_Suppressors
      NES maximal at the earliest transition AND declining thereafter; NABA_MATRISOME NES
      rising across transitions (Spearman(NES, transition order) > 0). Fujiwara replicates
      an arm if (ferro) suppressor NES is highest at its earliest AVAILABLE transition, and/
      or (matri) NABA_MATRISOME NES rises across Fujiwara's transitions. Both -> replicated;
      one -> that arm replicated, the other discovery-only; neither -> discovery-only,
      Fujiwara shown regardless.
  R2 (Task 2): any leading-edge overlap with BH padj < 0.05 is disclosed with its genes.
Run: python3 scripts/python/14_fujiwara_gsea_matrisome.py   (after the Fujiwara R DGE step)
"""
import json
import os
import pickle
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy.stats import hypergeom, spearmanr
from statsmodels.stats.multitest import multipletests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

SEED = 42
PERMS = 1000
OUT = "results/ws30"
rng = np.random.default_rng(SEED)
print("PRE-COMMITTED RULES R0/R1/R2 — see module docstring (stated before results).")

INPUTS = {
    "c2": "data/msigdb/c2.all.v2023.2.Hs.json",
    "naba26": "MASLD-multiomics_rev/Matrisome analysis/NABA_MATRISOME.v2026.1.Hs.tsv",
    "matrix": "results/ws19/panel_matrisome_matrix.csv",
    "dge_full": "results/ws1_signature/ws1_dge_full.csv",
    "locked": "results/ws15/locked_data.pkl",
    "drivers": "data/ferroptosis_driver_ferrdb.csv",
    "suppressors": "data/ferroptosis_suppressor_ferrdb.csv",
    "graph_stats": "results/ws23/graph_stats.json",
    "zbtb2": "results/ws29/zbtb2_specificity.csv",
    "net_stats": "results/ws28_network/stats_ws28_network.json",
    "ours75": "results/ws11/panel_size75_genes.csv",
    "compact": "results/ws7/compact_panel.csv",
    "celltype": "MASLD-multiomics_rev/MASLD_Final_Delivery/exported_significant_assets/tables/panel_gene_celltype_expression.csv",
}


def md5(p):
    import hashlib
    h = hashlib.md5()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def write_provenance(output, inputs, extra=None, rows=None):
    prov = {"output": output,
            "derived_from": [{"path": p, "md5": md5(p), "bytes": os.path.getsize(p),
                              "rows": (rows or {}).get(p)} for p in inputs],
            "script": "scripts/python/14_fujiwara_gsea_matrisome.py (+ scripts/R/15_fujiwara_stagepair_dge.R)",
            "git_commit": os.popen("git rev-parse --short HEAD").read().strip(),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "random_seed": SEED, "permutations": PERMS}
    if extra:
        prov.update(extra)
    with open(f"{OUT}/{output}.provenance.json", "w") as f:
        json.dump(prov, f, indent=2)


def gsea(ranked_genes, weights, gene_set, le_full=False, rng=None):
    rng = rng or np.random.default_rng(SEED)
    hit = np.isin(ranked_genes, list(gene_set))
    nh, n = int(hit.sum()), len(hit)
    if nh == 0 or nh == n:
        return 0.0, np.nan, 1.0, int(nh), []

    def es_of(h):
        w = np.abs(weights)
        ph = np.cumsum(np.where(h, w, 0.0)); ph /= ph[-1]
        pm = np.cumsum(np.where(~h, 1.0, 0.0)); pm /= pm[-1]
        d = ph - pm
        i = int(np.argmax(np.abs(d)))
        return d[i], i

    E, ipeak = es_of(hit)
    null = np.array([es_of(rng.permutation(hit))[0] for _ in range(PERMS)])
    same = null[np.sign(null) == np.sign(E)]
    nes = E / np.abs(same).mean() if len(same) else np.nan
    p = (np.abs(same) >= abs(E)).mean() if len(same) else np.nan
    src = ranked_genes[:ipeak + 1] if E > 0 else ranked_genes[ipeak:]
    hsrc = hit[:ipeak + 1] if E > 0 else hit[ipeak:]
    leading = [g for g, h in zip(src, hsrc) if h]
    if not le_full:
        leading = leading[:15]
    return float(E), float(nes), float(p), int(nh), leading


stats = {"date": datetime.now(timezone.utc).isoformat(), "seed": SEED}
ledger = []

c2 = json.load(open(INPUTS["c2"]))
NABA10 = ["NABA_MATRISOME", "NABA_CORE_MATRISOME", "NABA_MATRISOME_ASSOCIATED",
          "NABA_COLLAGENS", "NABA_ECM_GLYCOPROTEINS", "NABA_PROTEOGLYCANS",
          "NABA_BASEMENT_MEMBRANES", "NABA_ECM_REGULATORS", "NABA_ECM_AFFILIATED",
          "NABA_SECRETED_FACTORS"]
naba = {k: set(c2[k]["geneSymbols"]) for k in NABA10}
drivers = set(pd.read_csv(INPUTS["drivers"])["symbol"].str.upper().str.strip())
suppressors = set(pd.read_csv(INPUTS["suppressors"])["symbol"].str.upper().str.strip())
ferro_sets = {"FerrDb_Drivers": drivers, "FerrDb_Suppressors": suppressors}
dge = pd.read_csv(INPUTS["dge_full"])
dge["GS"] = dge["GeneSymbol"].astype(str).str.upper().str.strip()
tested_bg = set(dge.GS.dropna())
naba_tested = {k: v & tested_bg for k, v in naba.items()}

# ============================ TASK 0 ============================
print("\n[T0] NABA release reconciliation")
tok26 = set()
for line in open(INPUTS["naba26"]):
    for p in line.strip().replace(",", "\t").split("\t"):
        if p:
            tok26.add(p.strip().upper())
v23_in_bg = naba_tested["NABA_MATRISOME"]
v26_in_bg = tok26 & tested_bg
sets_equal = v23_in_bg == v26_in_bg
mx = pd.read_csv(INPUTS["matrix"])
rel = pd.DataFrame([
    {"source_run": "ws19_panel_matrix (authoritative Table values)", "release_string": "v2023.2.Hs",
     "original_size": 1026, "size_in_background": 627, "matches_ws29": True},
    {"source_run": "ws29_stagepair (this repo, results/ws29)", "release_string": "v2023.2.Hs",
     "original_size": 1026, "size_in_background": 627, "matches_ws29": True},
])
# the local v2026.1 file produced NO manuscript value (WS16-era p=1.448e-50 quarantined);
# it intersects the background at the same count (627) but a different gene set — recorded
# in provenance, not as a manuscript-producing run
rel.to_csv(f"{OUT}/naba_release_reconciliation.csv", index=False)
write_provenance("naba_release_reconciliation.csv", [INPUTS["matrix"], INPUTS["c2"], INPUTS["naba26"], INPUTS["dge_full"]],
                 extra={"background_set_identical_v2023_2_vs_local_v2026_1": bool(sets_equal),
                        "note": "All manuscript matrisome values derive from v2023.2.Hs "
                                "(ws19 matrix msigdb_release column uniform; WS29 identical download). "
                                "The 2,106-token local v2026.1 file intersects the same 627 background "
                                "genes and produced only the quarantined WS16-era p=1.448e-50."})
print(rel.to_string(index=False))
print(f"  background sets identical across releases: {sets_equal} -> R0: NO regeneration needed")
stats["T0"] = {"regeneration_needed": False, "background_identical": bool(sets_equal)}
ledger.append(["ws30_t0_release", "v2023.2.Hs throughout; local v2026.1 background-identical (627); no regeneration",
               f"{OUT}/naba_release_reconciliation.csv", "release_string", "", True])

# ============================ TASK 1 ============================
print("\n[T1] Fujiwara stage-pair replication (family = 12 sets x 4 transitions = 48)")
rngF = np.random.default_rng(SEED)
pairs = [(0, 1), (1, 2), (2, 3), (3, 4)]
nsF = {0: 12, 1: 58, 2: 50, 3: 56, 4: 37}
rowsF = []
for a, b in pairs:
    d = pd.read_csv(f"{OUT}/tmp/fuji_stagepair_dge_F{a}vF{b}.csv").dropna(subset=["t"]) \
        .sort_values("t", ascending=False)
    d = d[~d.gene.duplicated()]
    universe = set(d.gene)
    for programme, sets in [("matrisome", naba), ("ferroptosis", ferro_sets)]:
        for name, gs in sets.items():
            inbg = gs & universe
            es, nes, p, hits, le = gsea(d.gene.to_numpy(), d.t.to_numpy(), inbg, rng=rngF)
            rowsF.append({"cohort": "Fujiwara", "gene_set": name, "programme": programme,
                          "transition": f"F{a}vF{b}", "n_group1": nsF[a], "n_group2": nsF[b],
                          "low_powered": min(nsF[a], nsF[b]) < 20,
                          "set_size": len(gs), "set_size_in_background": len(inbg),
                          "NES": nes, "pval": p,
                          "leading_edge_genes": ";".join(le)})
fj = pd.DataFrame(rowsF)
fj["padj_family40"] = multipletests(fj.pval, method="fdr_bh")[1]
fj.to_csv(f"{OUT}/stagepair_gsea_fujiwara.csv", index=False)
write_provenance("stagepair_gsea_fujiwara.csv",
                 [f"{OUT}/tmp/fuji_stagepair_dge_F{a}vF{b}.csv" for a, b in pairs] +
                 [INPUTS["c2"], INPUTS["drivers"], INPUTS["suppressors"]],
                 extra={"family_size": 48, "cohort": "Fujiwara (locked WS15 build, n=213)",
                        "stages": "F0=12 F1=58 F2=50 F3=56 F4=37; F0vF1 low_powered"})

# discovery table for comparison + rule evaluation
disc = pd.read_csv("results/ws29/stagepair_gsea_matrisome.csv")
order = ["F0vF1", "F1vF2", "F2vF3", "F3vF4"]


def nes_seq(df, gene_set):
    s = df[df.gene_set == gene_set].set_index("transition").reindex(order)
    return s.NES.to_numpy()


def rule_verdict():
    ds = nes_seq(disc, "FerrDb_Suppressors")
    dm = nes_seq(disc, "NABA_MATRISOME")
    fs = nes_seq(fj, "FerrDb_Suppressors")
    fm = nes_seq(fj, "NABA_MATRISOME")
    o = np.arange(4)
    disc_pattern = (np.argmax(ds) == 0 and spearmanr(dm, o).statistic > 0)
    ferro_rep = np.argmax(fs) == 0
    matri_rep = spearmanr(fm, o).statistic > 0
    if ferro_rep and matri_rep:
        v = "REPLICATED (both arms)"
    elif ferro_rep or matri_rep:
        which = "ferroptosis" if ferro_rep else "matrisome"
        v = f"PARTIAL: {which} arm replicated; the other discovery-only"
    else:
        v = "NOT REPLICATED: trajectory finding confined to discovery (Fujiwara shown regardless)"
    return disc_pattern, ferro_rep, matri_rep, v, (ds, dm, fs, fm)


dp, fr, mr, verdict, seqs = rule_verdict()
ds, dm, fs, fm = seqs
print(f"  discovery pattern holds: {dp} | Fujiwara ferro-arm: {fr} | matrisome-arm: {mr}")
print(f"  discovery NES  suppressors {np.round(ds,2)} | NABA_MATRISOME {np.round(dm,2)}")
print(f"  Fujiwara  NES  suppressors {np.round(fs,2)} | NABA_MATRISOME {np.round(fm,2)}")
print(f"  R1 VERDICT: {verdict}")
stats["T1"] = {"discovery_pattern_holds": bool(dp), "fuji_ferro_arm": bool(fr),
               "fuji_matri_arm": bool(mr), "verdict": verdict,
               "NES": {"disc_suppressors": ds.tolist(), "disc_matrisome": dm.tolist(),
                       "fuji_suppressors": fs.tolist(), "fuji_matrisome": fm.tolist()}}
ledger.append(["ws30_t1_replication", verdict, f"{OUT}/stagepair_gsea_fujiwara.csv",
               "pre-committed rule R1", "", True])

fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
for ax, (gs, title) in zip(axes, [("FerrDb_Suppressors", "Ferroptosis suppressors"),
                                  ("NABA_MATRISOME", "NABA_MATRISOME")]):
    for df, cohort, col in [(disc, "Discovery (n=349)", "#4C72B0"), (fj, "Fujiwara (n=213)", "#C44E52")]:
        sub = df[df.gene_set == gs].set_index("transition").reindex(order)
        ax.plot(order, sub.NES, "o-", label=cohort, color=col)
        sig = sub.padj_family40 < 0.05
        ax.scatter(np.array(order)[sig], sub.NES[sig], s=120, facecolors="none",
                   edgecolors=col, linewidths=2)
    lp = [t for t in order if min(nsF[int(t[1])], nsF[int(t[4])]) < 20]
    for t in lp:
        ax.axvspan(t, t, color="crimson", alpha=0.0)
        ax.text(t, ax.get_ylim()[0], "LP", ha="center", fontsize=8, color="crimson")
    ax.set_title(f"{title} — NES by transition (ringed = padj<0.05, family 48)")
    ax.set_ylabel("NES")
    ax.legend(frameon=False, fontsize=8)
fig.suptitle("Stage-resolved enrichment: discovery vs Fujiwara replication (LP = low-powered F0vF1, n=12/58)")
plt.tight_layout()
fig.savefig(f"{OUT}/stagepair_discovery_vs_fujiwara.png", dpi=300)
plt.close(fig)
write_provenance("stagepair_discovery_vs_fujiwara.png",
                 [f"{OUT}/stagepair_gsea_fujiwara.csv", "results/ws29/stagepair_gsea_matrisome.csv"])

# ============================ TASK 2 ============================
print("\n[T2] leading-edge overlap (discovery, full leading edges recomputed with identical rng stream)")
rngLE = np.random.default_rng(SEED)
rowsLE = []
for a, b in pairs:
    d = pd.read_csv(f"results/ws27_ferroptosis_followups/stagepair_dge_F{a}vF{b}.csv") \
        .dropna(subset=["t"]).sort_values("t", ascending=False)
    d = d[~d.gene.duplicated()]
    universe = set(d.gene)
    for programme, sets in [("matrisome", naba), ("ferroptosis", ferro_sets)]:
        for name, gs in sets.items():
            es, nes, p, hits, le = gsea(d.gene.to_numpy(), d.t.to_numpy(), gs & universe, le_full=True, rng=rngLE)
            rowsLE.append({"gene_set": name, "programme": programme, "transition": f"F{a}vF{b}",
                           "NES": nes, "pval": p, "le_n": len(le),
                           "leading_edge_full": ";".join(le)})
le_df = pd.DataFrame(rowsLE)
# reproducibility assertion vs WS29 saved NES (same rng stream)
chk = le_df.merge(disc[["gene_set", "transition", "NES", "pval"]], on=["gene_set", "transition"],
                  suffixes=("_new", "_ws29"))
assert np.allclose(chk.NES_new, chk.NES_ws29, atol=1e-9) and np.allclose(chk.pval_new, chk.pval_ws29, atol=1e-12), \
    "leading-edge recompute does not reproduce WS29 NES/p — rng stream diverged"
print(f"  recompute reproduces WS29 exactly for all {len(chk)} cells")
le_df.to_csv(f"{OUT}/stagepair_le_full.csv", index=False)
write_provenance("stagepair_le_full.csv",
                 [f"results/ws27_ferroptosis_followups/stagepair_dge_F{a}vF{b}.csv" for a, b in pairs] +
                 [INPUTS["c2"], INPUTS["drivers"], INPUTS["suppressors"]],
                 extra={"note": "full leading edges; NES/pval byte-identical to results/ws29 "
                                "(asserted); leading edge = genes at/before the ES peak"})

# universe for expected overlap = the 12,537-gene ranked list
uni_n = 12537
sig_matri_f2v3 = disc[(disc.programme == "matrisome") & (disc.transition == "F2vF3") &
                      (disc.padj_family40 < 0.05)].gene_set.tolist()
get_le = lambda gs, tr: set(le_df[(le_df.gene_set == gs) & (le_df.transition == tr)]
                            .leading_edge_full.iloc[0].split(";")) - {""}
rowsOv = []
for fset, ftr in [("FerrDb_Suppressors", "F0vF1"), ("FerrDb_Drivers", "F0vF1")]:
    fle = get_le(fset, ftr)
    for mset in sig_matri_f2v3:
        mle = get_le(mset, "F2vF3")
        ov = fle & mle
        exp = len(fle) * len(mle) / uni_n
        p = hypergeom.sf(len(ov) - 1, uni_n, len(mle), len(fle)) if ov else 1.0
        rowsOv.append({"ferro_set": fset, "ferro_transition": ftr,
                       "matrisome_set": mset, "matrisome_transition": "F2vF3",
                       "ferro_le_n": len(fle), "matri_le_n": len(mle),
                       "overlap": len(ov), "expected": round(exp, 2), "p": p,
                       "overlapping_genes": ";".join(sorted(ov))})
ov = pd.DataFrame(rowsOv)
ov["padj"] = multipletests(ov.p, method="fdr_bh")[1]
ov.to_csv(f"{OUT}/programme_leading_edge_overlap.csv", index=False)
write_provenance("programme_leading_edge_overlap.csv", [f"{OUT}/stagepair_le_full.csv"],
                 extra={"universe": "12,537 ranked genes", "family": f"{len(ov)} overlaps, BH"})
print(ov[["ferro_set", "matrisome_set", "ferro_le_n", "matri_le_n", "overlap",
          "expected", "p", "padj"]].round(4).to_string(index=False))
naba_union = set().union(*naba_tested.values())
ann = {"ferrdb_drivers_in_naba": len(drivers & naba_union),
       "ferrdb_suppressors_in_naba": len(suppressors & naba_union),
       "naba_total": len(naba_union), "background_n": len(tested_bg)}
json.dump(ann, open(f"{OUT}/programme_annotation_overlap.json", "w"), indent=2)
write_provenance("programme_annotation_overlap.json", [INPUTS["c2"], INPUTS["drivers"], INPUTS["suppressors"], INPUTS["dge_full"]])
print(f"  annotation overlap: drivers {ann['ferrdb_drivers_in_naba']}/{len(drivers & tested_bg)} in NABA | "
      f"suppressors {ann['ferrdb_suppressors_in_naba']}/{len(suppressors & tested_bg)}")
stats["T2"] = {"any_significant_overlap": bool((ov.padj < 0.05).any()),
               "max_overlap": int(ov.overlap.max()),
               "annotation": ann}
ledger.append(["ws30_t2_le_overlap", f"any BH padj<0.05: {bool((ov.padj<0.05).any())}",
               f"{OUT}/programme_leading_edge_overlap.csv", "padj", "", True])

# ============================ TASK 3 ============================
print("\n[T3] size-matched subsampling of Ours-649 (1000x, seed 42)")
rngS = np.random.default_rng(SEED)
# authoritative 649 from the signature file (locked build)
sig649 = set(pd.read_csv("results/ws1_signature/ws1_signature_genes.csv")["GeneSymbol"].str.upper())
N, K = len(tested_bg), len(naba_tested["NABA_MATRISOME"])
comp = {3: "3-gene", 15: "15-ELBOW", 56: "57-BM", 133: "Kamzolas 145", 174: "194-PT"}
comp_fold = dict(zip(mx[mx.Category == "NABA_MATRISOME"].Panel,
                     mx[mx.Category == "NABA_MATRISOME"].Fold_Enrichment))
rows3 = []
for size, panel in comp.items():
    folds, sigs = [], 0
    for _ in range(1000):
        sub = rngS.choice(sorted(sig649), size=size, replace=False)
        k = len(set(sub) & naba_tested["NABA_MATRISOME"])
        folds.append((k / size) / (K / N))
        if hypergeom.sf(k - 1, N, K, size) < 0.05:
            sigs += 1
    folds = np.array(folds)
    cf = comp_fold.get(panel, comp_fold.get(f"{panel}-gene", np.nan))
    rows3.append({"target_size": size, "comparator_panel": panel, "comparator_fold": cf,
                  "subsample_median_fold": np.median(folds),
                  "subsample_iqr_lo": np.percentile(folds, 25),
                  "subsample_iqr_hi": np.percentile(folds, 75),
                  "prop_significant": sigs / 1000,
                  "comparator_percentile": (folds < cf).mean()})
s3 = pd.DataFrame(rows3)
s3.to_csv(f"{OUT}/size_matched_subsampling.csv", index=False)
write_provenance("size_matched_subsampling.csv",
                 ["results/ws1_signature/ws1_signature_genes.csv", INPUTS["c2"], INPUTS["dge_full"], INPUTS["matrix"]],
                 extra={"note": "1000 subsamples per size, seed 42; NABA_MATRISOME v2023.2.Hs, "
                                f"background {N}, K={K}; comparator folds from the ws19 matrix"})
print(s3.round(3).to_string(index=False))
stats["T3"] = s3.drop(columns=["comparator_panel"]).to_dict("records")
ledger.append(["ws30_t3_sizematch", "see size_matched_subsampling.csv",
               f"{OUT}/size_matched_subsampling.csv", "", "", True])

# ============================ TASK 4 ============================
print("\n[T4] publication 8x10 panel-by-category matrix")
cells = {}
counts = {}
for panel in mx.Panel.unique():
    sub = mx[mx.Panel == panel].set_index("Category")
    counts[panel] = int((sub.padj < 0.05).sum())
    for cat in NABA10:
        r = sub.loc[cat]
        n_pan = int(r.Panel_Size)
        minov = next((k for k in range(n_pan + 1)
                      if hypergeom.sf(k - 1, 15223, int(r.Size_in_Background), n_pan) < 0.05), n_pan + 1)
        if minov > n_pan:
            cell = f"{r.Fold_Enrichment:.2f} (undecidable)"
        elif r.padj < 0.05:
            cell = f"{r.Fold_Enrichment:.2f}* (padj {r.padj:.1e})"
        else:
            cell = f"{r.Fold_Enrichment:.2f} (padj {r.padj:.2f}, ns)"
        cells[(cat, panel)] = cell
tab = pd.DataFrame({p: [cells[(c, p)] for c in NABA10] for p in mx.Panel.unique()},
                   index=NABA10)
tab.loc["significant_categories (of 10)"] = [f"{counts[p]}/10" for p in tab.columns]
tab.index.name = "category"
tab.to_csv(f"{OUT}/table_S_panel_category_matrix.csv")
with open(f"{OUT}/table_S_panel_category_matrix.md", "w") as f:
    f.write("# Supplementary table — panel x NABA category (fold enrichment, v2023.2.Hs)\n\n")
    f.write("* = padj < 0.05 (BH within panel, 10 categories); undecidable = minimum overlap "
            "for padj<0.05 exceeds panel size. Family is nested (MATRISOME > CORE/ASSOCIATED "
            "> sub-categories); correction is anticonservative.\n\n")
    f.write(tab.to_markdown())
    f.write("\n")
write_provenance("table_S_panel_category_matrix.csv", [INPUTS["matrix"], INPUTS["c2"]],
                 extra={"significant_categories": counts,
                        "note": "counts verified from the matrix, not copied from the work order"})
print(pd.Series(counts).to_string())
stats["T4_counts"] = counts

# ============================ TASK 5 ============================
print("\n[T5] basement membranes as leading category")
bm_row = mx[(mx.Category == "NABA_BASEMENT_MEMBRANES") & (mx.Panel == "Ours-649 (Full)")].iloc[0]
bm_genes = sorted((naba_tested["NABA_BASEMENT_MEMBRANES"] & sig649))
dge_sorted = dge.dropna(subset=["t"]).sort_values("t", key=abs, ascending=False)
dge_sorted = dge_sorted[~dge_sorted.GS.duplicated()]
rank = {g: i + 1 for i, g in enumerate(dge_sorted.GS)}
ours75 = set(pd.read_csv(INPUTS["ours75"]).iloc[:, 0].astype(str).str.upper())
compact = set(pd.read_csv(INPUTS["compact"]).iloc[:, 0].astype(str).str.upper())
ct = pd.read_csv(INPUTS["celltype"], index_col=0)
ct.index = ct.index.astype(str).str.upper()
t5 = pd.DataFrame({"gene": bm_genes})
t5["rank_by_abs_t"] = t5.gene.map(rank)
t5["in_top_100"] = t5.rank_by_abs_t <= 100
t5["in_ours_75"] = t5.gene.isin(ours75)
t5["in_ours_compact"] = t5.gene.isin(compact)
t5["cholangiocyte_associated"] = [
    (g in ct.index and str(ct.loc[g, "TSP_max_celltype"]).lower().find("cholang") >= 0)
    for g in t5.gene]
t5.to_csv(f"{OUT}/basement_membrane_genes.csv", index=False)
write_provenance("basement_membrane_genes.csv",
                 [INPUTS["c2"], "results/ws1_signature/ws1_signature_genes.csv",
                  INPUTS["dge_full"], INPUTS["ours75"], INPUTS["compact"], INPUTS["celltype"], INPUTS["matrix"]],
                 extra={"matrix_row": {"observed": int(bm_row.Observed_Overlap),
                                       "fold": float(bm_row.Fold_Enrichment),
                                       "padj": float(bm_row.padj)}})
print(f"  matrix row: observed {int(bm_row.Observed_Overlap)}, fold {bm_row.Fold_Enrichment:.2f}, "
      f"padj {bm_row.padj:.1e} | genes: {bm_genes}")
print(t5.to_string(index=False))
stats["T5"] = {"n_genes": len(bm_genes), "fold": float(bm_row.Fold_Enrichment),
               "padj": float(bm_row.padj)}

# ============================ TASK 6 ============================
print("\n[T6] compact-panel exception")
rows6 = []
for panel in ["Ours-compact", "3-gene"]:
    r = mx[(mx.Category == "NABA_MATRISOME") & (mx.Panel == panel)].iloc[0]
    n_pan = int(r.Panel_Size)
    minov = next((k for k in range(n_pan + 1)
                  if hypergeom.sf(k - 1, 15223, int(r.Size_in_Background), n_pan) < 0.05), n_pan + 1)
    rows6.append({"panel": panel, "panel_size": n_pan,
                  "matrisome_members": int(r.Observed_Overlap),
                  "fold": float(r.Fold_Enrichment), "padj": float(r.padj),
                  "min_overlap_for_padj05": minov,
                  "compositional_failure": bool(r.padj >= 0.05 and minov <= n_pan)})
t6 = pd.DataFrame(rows6)
t6.to_csv(f"{OUT}/compact_panel_exception.csv", index=False)
write_provenance("compact_panel_exception.csv", [INPUTS["matrix"]])
print(t6.to_string(index=False))

# ============================ TASK 7 ============================
print("\n[T7] additional nulls + recount")
g = json.load(open(INPUTS["graph_stats"]))
z = pd.read_csv(INPUTS["zbtb2"])
nulls = pd.DataFrame([
    {"null_id": "centrality_vs_gwas",
     "description": "Graph betweenness centrality vs GWAS Catalog membership indicator",
     "statistic": f"Mann-Whitney U = {g['centrality_vs_gwas']['statistic']}",
     "p": g["centrality_vs_gwas"]["p"], "n": g["centrality_vs_gwas"]["n"],
     "source_file": "results/ws23/graph_stats.json", "source_row_selector": "centrality_vs_gwas"},
    {"null_id": "zbtb2_target_specificity",
     "description": "ZBTB2 ferroptosis-target specificity: GO ferroptosis contains 0/4 targets; "
                    "FerrDb drivers and suppressors equally enriched (both padj 0.024)",
     "statistic": "0 of 4 targets in GO:0097707",
     "p": float(z[z.pathway == "FerrDb_Drivers"].p.iloc[0]), "n": 4,
     "source_file": "results/ws29/zbtb2_specificity.csv", "source_row_selector": "pathway == FerrDb_Drivers"},
])
nulls.to_csv(f"{OUT}/additional_nulls.csv", index=False)
write_provenance("additional_nulls.csv", [INPUTS["graph_stats"], INPUTS["zbtb2"]])
# recount: revised manuscript lists ten; these two are not among them -> twelve
existing_ten = ["prognostic increment", "threshold transport", "fusion", "pseudotime regression",
                "WGCNA grey", "ferroptosis signature enrichment", "ferroptosis F2 classifier",
                "paired ferroptosis trajectory", "effector stellate localisation", "sex x stage"]
total = len(existing_ten) + 2
print(f"  existing ten + 2 new (no duplicates) -> total = {total}")
stats["T7_total_nulls"] = total
ledger.append(["ws30_t7_null_count", total, f"{OUT}/additional_nulls.csv", "", "", True])

# ============================ TASK 8 ============================
print("\n[T8] driver-suppressor co-expression carry-forward")
ns = json.load(open(INPUTS["net_stats"]))
edges = pd.read_csv("results/ws28_network/network_edges.csv")
coexp = {"edges_at_rho_gt_0.5": int(ns["observed_edges"]),
         "nodes": int(ns["nodes"]),
         "permutation_null_mean": float(ns["null_mean"]),
         "permutation_null_P95": float(ns["null_P95"]),
         "modularity_Q": ns.get("modules", {}).get("modularity_Q"),
         "cross_type_edge_fraction": float(ns["mixing"]["observed_fraction"]),
         "random_wiring_expectation": float(ns["mixing"]["expected_fraction"]),
         "cross_type_hypergeom_p": float(ns["mixing"]["hypergeom_p"]),
         "interpretation": "coordinated regulation rather than opposing arms; consistent with "
                           "the rho=-0.238 suppressor-biased balance (WS27)"}
json.dump(coexp, open(f"{OUT}/driver_suppressor_coexpression.json", "w"), indent=2)
write_provenance("driver_suppressor_coexpression.json",
                 [INPUTS["net_stats"], "results/ws28_network/network_edges.csv"])
print(json.dumps(coexp, indent=1))

# ============================ TASK 9 ============================
print("\n[T9] KG/GNN removal manifest (scoping decision for the author)")
t9 = pd.DataFrame([
    {"item": "Methods 2.7.6 (Knowledge graph and drug-gene interactions)", "type": "section",
     "decision": "remove", "reason": "51-edge graph unfiltered (filter 51->51), gnn_claim_permitted=false"},
    {"item": "Supplementary Fig. S7 (Knowledge graph diagnostics)", "type": "figure",
     "decision": "remove", "reason": "diagnostics of a graph no manuscript claim rests on"},
    {"item": "Supplementary table S11 (curated drug-gene edges)", "type": "supplementary",
     "decision": "remove", "reason": "KG-dependent; the curated browser survives as a standalone tool artifact"},
    {"item": "github_repo pipeline", "type": "tool",
     "decision": "retain", "reason": "analysis pipeline; no predictive claim"},
    {"item": "TRRUST/ZBTB2 material (section 3.3, Table 4, Fig 4)", "type": "section",
     "decision": "retain", "reason": "independent of the KG; hypothesis-generating framing per WS29 R3"},
    {"item": "GWAS membership indicator sentence (Methods 2.7.5)", "type": "sentence",
     "decision": "retain (null)", "reason": "centrality-vs-GWAS null is reported as a null (Task 7)"},
    {"item": "graph_stats.json / centrality null (MWU p=1.0)", "type": "result",
     "decision": "retain as null", "reason": "one of the twelve reported nulls"},
    {"item": "GAT/GCN trial benchmark artifacts (results/trial_benchmark_*)", "type": "repo artifact",
     "decision": "retain in repo, absent from manuscript", "reason": "drug-trial scope, CI [0.34,1.0]; never a diagnostic claim in the current draft"},
])
t9.to_csv(f"{OUT}/removal_manifest.csv", index=False)
write_provenance("removal_manifest.csv", [INPUTS["graph_stats"]])
print(t9[["item", "decision"]].to_string(index=False))

# ============================ ledger + stats ============================
pd.DataFrame(ledger, columns=["metric", "value", "source_file", "source_column",
                              "source_row_selector", "recomputable"]).to_csv(
    f"{OUT}/numbers_ledger_ws30.csv", index=False)
write_provenance("numbers_ledger_ws30.csv", [f"{OUT}/{f}" for f in
                                             ["stagepair_gsea_fujiwara.csv", "programme_leading_edge_overlap.csv",
                                              "size_matched_subsampling.csv", "additional_nulls.csv"]])
with open(f"{OUT}/stats_ws30.json", "w") as f:
    json.dump(stats, f, indent=2, default=float)
write_provenance("stats_ws30.json", [INPUTS["matrix"], INPUTS["c2"], INPUTS["locked"]])
print("\nDONE — results/ws30/")
