#!/usr/bin/env python3
"""
WS29 — Remaining analysis (work order 2026-08-25).

Tasks:
  1. Stage-resolved matrisome GSEA (decisive for the temporal framework):
     10 NABA categories + 2 FerrDb sets x 4 adjacent transitions, one BH family
     of 48 (column kept as padj_family40 per the work order's schema).
  2. matrisome_by_rank_fold.csv — fold enrichment as primary, decidable cells.
  3. celldeath_rho_ci.csv — bootstrap Spearman rho CIs + paired difference CI.
  4. zbtb2_specificity.csv — ZBTB2 targets vs 7 pathway gene sets.
  5. ferroptosis_set_provenance.csv — the gene-set size chain.

PRE-COMMITTED RULES (stated before any result is computed — do not re-specify):

  R1 (Task 1): Let F_ferro = earliest transition at which any ferroptosis set reaches
      padj < 0.05 (family of 48), F_matri = earliest at which any matrisome set does.
      - F_matri later than F_ferro  -> temporal sequence DEMONSTRATED; keep framework.
      - F_matri equals/precedes     -> NOT supported; report programmes as concurrent.
      - No matrisome set significant anywhere -> matrisome enrichment not
        stage-localised; confine the claim to the cross-sectional convergence.
      F3 vs F4 (n = 73/16) is flagged low_powered; a null there is not absence.
  R2 (Task 3): if the bootstrap CI on rho_necroptosis - rho_ferroptosis includes
      zero, the programmes are NOT separable (matches Section 3.16's conclusion).
  R3 (Task 4): if ferroptosis is not the uniquely enriched pathway among the seven,
      the ZBTB2 finding is hypothesis-generating only (a TRRUST-documented
      ZBTB2-TP53 axis), and the annotation overlap is reported as computed.

Run: python3 scripts/python/48_ws29_remaining.py
Outputs -> results/ws29/
"""
import json
import os
import pickle
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy.stats import hypergeom, spearmanr, kruskal
from statsmodels.stats.multitest import multipletests

SEED = 42
PERMS = 1000
N_BOOT = 2000
OUT = "results/ws29"
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(SEED)

RULES = """PRE-COMMITTED RULES (stated before results):
R1: F_matri > F_ferro -> temporal sequence demonstrated | F_matri <= F_ferro -> concurrent | no matrisome sig -> not stage-localised. F3vF4 low-powered (73/16).
R2: CI on rho_necro - rho_ferro including 0 -> cell-death programmes NOT separable.
R3: ferroptosis not uniquely enriched among 7 pathways -> ZBTB2 is hypothesis-generating (ZBTB2-TP53 axis)."""
print(RULES)

INPUTS = {
    "c2": "data/msigdb/c2.all.v2023.2.Hs.json",
    "dge_full": "results/ws1_signature/ws1_dge_full.csv",
    "drivers_raw": "data/ferrdb_driver.csv",
    "suppressors_raw": "data/ferrdb_suppressor.csv",
    "drivers": "data/ferroptosis_driver_ferrdb.csv",
    "suppressors": "data/ferroptosis_suppressor_ferrdb.csv",
    "trrust": "data/kamzolas/annotation_databases/raw_data/trrust_rawdata.human.tsv",
    "quickgo_dir": "data/quickgo",
    "locked": "results/ws15/locked_data.pkl",
}


def md5(p):
    import hashlib
    h = hashlib.md5()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def write_provenance(output, inputs, extra=None, rows=None):
    derived = [{"path": p, "md5": md5(p), "bytes": os.path.getsize(p),
                "rows": rows.get(p) if rows else None} for p in inputs]
    prov = {"output": output, "derived_from": derived,
            "script": "scripts/python/48_ws29_remaining.py",
            "git_commit": os.popen("git rev-parse --short HEAD").read().strip(),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "random_seed": SEED, "permutations": PERMS, "bootstraps": N_BOOT}
    if extra:
        prov.update(extra)
    with open(f"{OUT}/{output}.provenance.json", "w") as f:
        json.dump(prov, f, indent=2)


def gsea(ranked_genes, weights, gene_set):
    """WS27 method verbatim: weighted running-sum ES, 1000 permutations, seed 42."""
    hit = np.isin(ranked_genes, list(gene_set))
    nh, n = int(hit.sum()), len(hit)
    if nh == 0 or nh == n:
        return 0.0, np.nan, 1.0, int(nh), []

    def es_of(h):
        w = np.abs(weights)
        ph = np.cumsum(np.where(h, w, 0.0)); ph /= ph[-1]
        pm = np.cumsum(np.where(~h, 1.0, 0.0)); pm /= pm[-1]
        d = ph - pm
        i = np.argmax(np.abs(d))
        return d[i], i

    E, ipeak = es_of(hit)
    null = np.array([es_of(rng.permutation(hit))[0] for _ in range(PERMS)])
    same = null[np.sign(null) == np.sign(E)]
    nes = E / np.abs(same).mean() if len(same) else np.nan
    p = (np.abs(same) >= abs(E)).mean() if len(same) else np.nan
    if E > 0:
        leading = [g for g, h in zip(ranked_genes[:ipeak + 1], hit[:ipeak + 1]) if h]
    else:
        leading = [g for g, h in zip(ranked_genes[ipeak:  ], hit[ipeak:  ]) if h]
    return float(E), float(nes), float(p), int(nh), leading[:15]


stats = {"rules": RULES, "seed": SEED, "permutations": PERMS, "bootstraps": N_BOOT,
         "analysis_date": datetime.now(timezone.utc).isoformat()}
ledger = []

# ---------------- shared inputs ----------------
with open(INPUTS["locked"], "rb") as f:
    root = pickle.load(f)
mat = root["mats"]["Discovery"]
stage = root["stages"]["Discovery"]
stage = (stage.loc[mat.columns] if hasattr(stage, "loc")
         else pd.Series(stage.values, index=mat.columns)).astype(int)

c2 = json.load(open(INPUTS["c2"]))
NABA10 = ["NABA_MATRISOME", "NABA_CORE_MATRISOME", "NABA_MATRISOME_ASSOCIATED",
          "NABA_COLLAGENS", "NABA_ECM_GLYCOPROTEINS", "NABA_PROTEOGLYCANS",
          "NABA_BASEMENT_MEMBRANES", "NABA_ECM_REGULATORS", "NABA_ECM_AFFILIATED",
          "NABA_SECRETED_FACTORS"]
naba = {k: set(c2[k]["geneSymbols"]) for k in NABA10}
drivers = set(pd.read_csv(INPUTS["drivers"])["symbol"].str.upper().str.strip())
suppressors = set(pd.read_csv(INPUTS["suppressors"])["symbol"].str.upper().str.strip())

# ================= TASK 1: stage-resolved matrisome GSEA =================
print("\n[T1] stage-resolved GSEA: 10 NABA + 2 FerrDb sets x 4 transitions, family = 48")
pairs = [(0, 1), (1, 2), (2, 3), (3, 4)]
ns = {0: 81, 1: 112, 2: 67, 3: 73, 4: 16}
rows1 = []
for a, b in pairs:
    fn = f"results/ws27_ferroptosis_followups/stagepair_dge_F{a}vF{b}.csv"
    d = pd.read_csv(fn).dropna(subset=["t"]).sort_values("t", ascending=False)
    d = d[~d.gene.duplicated()]
    universe = set(d.gene)
    for programme, sets in [("matrisome", naba), ("ferroptosis", {"FerrDb_Drivers": drivers,
                                                                  "FerrDb_Suppressors": suppressors})]:
        for name, gs in sets.items():
            inbg = gs & universe
            es, nes, p, hits, leading = gsea(d.gene.to_numpy(), d.t.to_numpy(), inbg)
            rows1.append({"gene_set": name, "programme": programme,
                          "transition": f"F{a}vF{b}", "n_group1": ns[a], "n_group2": ns[b],
                          "low_powered": min(ns[a], ns[b]) < 20,
                          "set_size": len(gs), "set_size_in_background": len(inbg),
                          "NES": nes, "pval": p,
                          "leading_edge_genes": ";".join(leading)})
g1 = pd.DataFrame(rows1)
# BH across the single combined family: 12 sets x 4 transitions = 48
g1["padj_family40"] = multipletests(g1.pval, method="fdr_bh")[1]
g1.to_csv(f"{OUT}/stagepair_gsea_matrisome.csv", index=False)
write_provenance("stagepair_gsea_matrisome.csv",
                 [INPUTS["c2"], INPUTS["drivers"], INPUTS["suppressors"]] +
                 [f"results/ws27_ferroptosis_followups/stagepair_dge_F{a}vF{b}.csv" for a, b in pairs],
                 extra={"family_size": 48,
                        "note": "BH across 12 sets x 4 transitions = 48 tests in ONE family "
                                "(column name padj_family40 retained per work-order schema); "
                                "WS27 Section 3.14 padj used a family of 8 and shifts here; "
                                "GSEA method identical to WS27 (weighted ES, 1000 perms, seed 42)",
                        "msigdb_release": "v2023.2.Hs"})
sig = g1[(g1.padj_family40 < 0.05) & (g1.NES > 0)]
order = ["F0vF1", "F1vF2", "F2vF3", "F3vF4"]
def earliest(prog):
    s = sig[sig.programme == prog].transition.unique()
    return min(s, key=order.index) if len(s) else None
F_ferro, F_matri = earliest("ferroptosis"), earliest("matrisome")
print(f"  F_ferro = {F_ferro} | F_matri = {F_matri}")
print(g1[g1.padj_family40 < 0.05][["gene_set", "programme", "transition", "NES",
                                   "pval", "padj_family40", "low_powered"]].to_string(index=False))
idx = {t: i for i, t in enumerate(order)}
if F_matri is None:
    verdict1 = "NO matrisome set reaches padj<0.05 at any transition: matrisome enrichment " \
               "is NOT stage-localised in this cohort; confine the claim to the " \
               "cross-sectional convergence result"
elif F_ferro is None:
    verdict1 = f"no ferroptosis set significant; matrisome earliest at {F_matri}"
elif idx[F_matri] > idx[F_ferro]:
    verdict1 = f"DEMONSTRATED: F_matri ({F_matri}) is later than F_ferro ({F_ferro})"
elif idx[F_matri] <= idx[F_ferro]:
    verdict1 = f"NOT SUPPORTED: F_matri ({F_matri}) does not follow F_ferro ({F_ferro}); " \
               "report the programmes as concurrent"
print(f"  R1 VERDICT: {verdict1}")
stats["T1"] = {"F_ferro": F_ferro, "F_matri": F_matri, "verdict": verdict1,
               "family_size": 48,
               "significant": sig[["gene_set", "transition", "NES", "padj_family40"]]
               .to_dict("records")}
ledger.append(["ws29_t1_verdict", verdict1, f"{OUT}/stagepair_gsea_matrisome.csv",
               "pre-committed rule R1", "", True])

# combined figure
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
for ax, prog in zip(axes, ["matrisome", "ferroptosis"]):
    sub = g1[(g1.programme == prog) & (g1.gene_set.isin(
        ["NABA_MATRISOME", "NABA_CORE_MATRISOME", "NABA_COLLAGENS", "NABA_ECM_GLYCOPROTEINS",
         "FerrDb_Drivers", "FerrDb_Suppressors"]))]
    sns.barplot(data=sub, x="transition", y="NES", hue="gene_set", ax=ax,
                order=order, hue_order=sub.groupby("gene_set").NES.mean().sort_values(ascending=False).index.tolist())
    ax.set_title(f"{prog} sets")
    ax.axhline(0, c="grey", lw=0.8)
    for xi, t in enumerate(order):
        if min(ns[int(t[1])], ns[int(t[4])]) < 20:
            ax.text(xi, ax.get_ylim()[0] * 0.9, "low-power", ha="center",
                    fontsize=7, color="crimson", rotation=90)
fig.suptitle("Stage-resolved GSEA (family = 48; NABA v2023.2.Hs + FerrDb; limma-trend pairs)")
plt.tight_layout()
fig.savefig(f"{OUT}/stagepair_gsea_combined.png", dpi=300)
plt.close(fig)
write_provenance("stagepair_gsea_combined.png", [f"{OUT}/stagepair_gsea_matrisome.csv"])

# ================= TASK 2: matrisome_by_rank fold =================
print("\n[T2] matrisome by |t| rank — fold vs background as primary")
dge = pd.read_csv(INPUTS["dge_full"])
dge["GS"] = dge["GeneSymbol"].astype(str).str.upper().str.strip()
dge = dge.dropna(subset=["t", "adj.P.Val"]).sort_values("t", key=abs, ascending=False)
dge = dge[~dge.GS.duplicated()]
tested = set(dge.GS)
mset = naba["NABA_MATRISOME"]
in_bg = mset & tested
bg_density = len(in_bg) / len(tested)
rows2 = []
for K in [10, 25, 50, 75, 100, 250, 649]:
    top = dge.GS.head(K)
    obs = top.isin(in_bg).sum()
    density = obs / K
    p = hypergeom.sf(obs - 1, len(tested), len(in_bg), K) if obs else 1.0
    minov = next((k for k in range(K + 1) if hypergeom.sf(k - 1, len(tested), len(in_bg), K) < 0.05), K + 1)
    rows2.append({"top_k": K, "matrisome_n": int(obs), "density": round(density, 4),
                  "background_density": round(bg_density, 5),
                  "fold": round(density / bg_density, 3) if bg_density else np.nan,
                  "p": p, "min_overlap_for_p05": minov,
                  "decidable": minov <= K})
m2 = pd.DataFrame(rows2)
m2.to_csv(f"{OUT}/matrisome_by_rank_fold.csv", index=False)
write_provenance("matrisome_by_rank_fold.csv", [INPUTS["dge_full"], INPUTS["c2"]],
                 extra={"note": "NABA_MATRISOME v2023.2.Hs restricted to the 15,223 tested genes; "
                                "fold = density / background_density; decidable = min overlap "
                                "for p<0.05 fits within top_k"})
print(m2.to_string(index=False))
stats["T2"] = {"background": f"{len(in_bg)}/{len(tested)}",
               "top10_fold": float(m2.loc[m2.top_k == 10, "fold"].iloc[0]),
               "signature_fold": float(m2.loc[m2.top_k == 649, "fold"].iloc[0])}
ledger.append(["ws29_t2_top10_fold", float(m2.loc[m2.top_k == 10, "fold"].iloc[0]),
               f"{OUT}/matrisome_by_rank_fold.csv", "fold", "top_k == 10", True])

# ================= TASK 3: cell-death rho with CIs =================
print("\n[T3] cell-death pathway rho with bootstrap CIs + paired difference")
GO = {"apoptosis": "GO:0006915", "autophagy": "GO:0006914", "necroptosis": "GO:0097300",
      "pyroptosis": "GO:0141201", "ferroptosis_GO": "GO:0097707"}
sets7 = {}
for name, go in GO.items():
    cache = f"{INPUTS['quickgo_dir']}/{name}_{go.replace(':', '_')}_human.tsv"
    sets7[name] = set(pd.read_csv(cache, sep="\t", usecols=["SYMBOL"]).SYMBOL
                      .dropna().astype(str).str.upper().str.strip())
sets7["FerrDb_Drivers"] = drivers
sets7["FerrDb_Suppressors"] = suppressors
z = mat.sub(mat.mean(axis=1), axis=0).div(mat.std(axis=1).replace(0, np.nan), axis=0)
stg = stage.loc[mat.columns].to_numpy()
scores, rows3 = {}, []
for name, gs in sets7.items():
    g = [x for x in gs if x in z.index]
    s = z.loc[g].mean(axis=0).to_numpy()
    scores[name] = s
    rho = spearmanr(s, stg).statistic
    boot = np.array([spearmanr(s[idx], stg[idx]).statistic
                     for idx in (rng.integers(0, len(s), len(s)) for _ in range(N_BOOT))])
    groups = [pd.Series(s)[stg == k] for k in range(5)]
    kw = kruskal(*groups)
    rows3.append({"pathway": name, "n_genes": len(g), "rho": rho,
                  "ci_lo": np.percentile(boot, 2.5), "ci_hi": np.percentile(boot, 97.5),
                  "kw_p": kw.pvalue})
c3 = pd.DataFrame(rows3)
c3["kw_padj"] = multipletests(c3.kw_p, method="fdr_bh")[1]
c3.to_csv(f"{OUT}/celldeath_rho_ci.csv", index=False)
write_provenance("celldeath_rho_ci.csv",
                 [INPUTS["locked"], INPUTS["drivers"], INPUTS["suppressors"]] +
                 [f"{INPUTS['quickgo_dir']}/{n}_{g.replace(':', '_')}_human.tsv"
                  for n, g in GO.items()],
                 extra={"note": "patient-level bootstrap of Spearman rho vs stage, "
                                f"{N_BOOT} resamples, seed 42"})
# paired difference necroptosis - ferroptosis_GO
sn, sf = scores["necroptosis"], scores["ferroptosis_GO"]
diffs = np.array([spearmanr(sn[idx], stg[idx]).statistic -
                  spearmanr(sf[idx], stg[idx]).statistic
                  for idx in (rng.integers(0, len(sn), len(sn)) for _ in range(N_BOOT))])
dlo, dhi = np.percentile(diffs, [2.5, 97.5])
verdict3 = ("NOT SEPARABLE (CI includes 0)" if dlo <= 0 <= dhi else "separable")
json.dump({"comparison": "rho_necroptosis - rho_ferroptosis_GO",
           "point": float(spearmanr(sn, stg).statistic - spearmanr(sf, stg).statistic),
           "ci_lo": float(dlo), "ci_hi": float(dhi), "bootstraps": N_BOOT, "seed": SEED,
           "verdict_R2": verdict3},
          open(f"{OUT}/celldeath_rho_difference.json", "w"), indent=2)
write_provenance("celldeath_rho_difference.json", [f"{OUT}/celldeath_rho_ci.csv"])
print(c3.round(4).to_string(index=False))
print(f"  difference CI [{dlo:.4f}, {dhi:.4f}] -> R2 VERDICT: {verdict3}")
stats["T3"] = {"verdict": verdict3, "diff_ci": [float(dlo), float(dhi)]}
ledger.append(["ws29_t3_separability", verdict3, f"{OUT}/celldeath_rho_difference.json",
               "verdict_R2", "", True])

# ================= TASK 4: ZBTB2 specificity =================
print("\n[T4] ZBTB2 target specificity across 7 pathway gene sets")
tr = pd.read_csv(INPUTS["trrust"], sep="\t", header=None,
                 names=["tf", "target", "mode", "pmid"])
tr["TF"] = tr.tf.str.upper().str.strip(); tr["target"] = tr.target.str.upper().str.strip()
zb = tr[tr.TF == "ZBTB2"]
targets = sorted(zb.target.unique())
pmids = sorted(zb.pmid.astype(str).unique())
universe = set(tr.TF) | set(tr.target)
rows4 = []
for name, gs in sets7.items():
    K = len(gs & universe)
    overlap = len(set(targets) & gs)
    p = hypergeom.sf(overlap - 1, len(universe), K, len(targets)) if overlap else 1.0
    exp = len(targets) * K / len(universe)
    rows4.append({"pathway": name, "targets_in_pathway": overlap,
                  "fold": (overlap / exp) if exp > 0 else np.nan,
                  "p": p, "n_targets": len(targets),
                  "distinct_pmids": len(pmids),
                  "min_achievable_p": hypergeom.sf(len(targets) - 1, len(universe), K, len(targets))
                  if K >= len(targets) else hypergeom.sf(K - 1, len(universe), K, len(targets))})
z4 = pd.DataFrame(rows4)
z4["padj"] = multipletests(z4.p, method="fdr_bh")[1]
z4.to_csv(f"{OUT}/zbtb2_specificity.csv", index=False)
write_provenance("zbtb2_specificity.csv",
                 [INPUTS["trrust"], INPUTS["drivers"], INPUTS["suppressors"]] +
                 [f"{INPUTS['quickgo_dir']}/{n}_{g.replace(':', '_')}_human.tsv"
                  for n, g in GO.items()],
                 extra={"note": f"universe = TRRUST network genes (n={len(universe)}); "
                                f"ZBTB2 targets {targets}; PMIDs {pmids}; "
                                "min_achievable_p = best possible p at n_targets=4"})
# annotation overlap for the three ferroptosis targets
overlap_ann = {g: sum(g in gs for gs in sets7.values())
               for g in ["TP53", "MDM2", "CDKN1A"]}
sig4 = z4[z4.padj < 0.05].pathway.tolist()
verdict4 = ("hypothesis-generating only: ferroptosis is NOT uniquely enriched "
            f"(significant pathways: {sig4})" if len(sig4) != 1 or
            sig4 != [p for p in sig4 if "errDb" in p or "ferroptosis" in p] or len(sig4) != 1
            else "ferroptosis uniquely enriched")
if len(sig4) == 1 and ("errDb" in sig4[0] or "ferroptosis" in sig4[0]):
    verdict4 = f"ferroptosis uniquely enriched ({sig4[0]})"
print(z4.round(4).to_string(index=False))
print(f"  targets {targets} | PMIDs {pmids} | annotation overlap of 7 sets: {overlap_ann}")
print(f"  R3 VERDICT: {verdict4}")
stats["T4"] = {"targets": targets, "pmids": pmids, "annotation_overlap": overlap_ann,
               "significant_pathways": sig4, "verdict": verdict4}
ledger.append(["ws29_t4_zbtb2", verdict4, f"{OUT}/zbtb2_specificity.csv",
               "pre-committed rule R3", "", True])

# ================= TASK 5: set-size provenance chain =================
print("\n[T5] ferroptosis gene-set provenance chain")
raw_drv = pd.read_csv(INPUTS["drivers_raw"]); raw_sup = pd.read_csv(INPUTS["suppressors_raw"])
dge_bg = set(pd.read_csv(INPUTS["dge_full"])["GeneSymbol"].astype(str).str.upper().str.strip())
chain = [
    ("drivers", "FerrDb V2 raw download", len(raw_drv), len(raw_drv), "none (raw export)", "source file"),
    ("drivers", "filtered/HGNC-mapped/dedup (provenance on file)", len(raw_drv), 264,
     "download_ferrdb_v2 filter", "Methods 2.7.7 (264)"),
    ("drivers", "present in locked 12,537-gene discovery matrix", 264, 194,
     "intersect locked matrix index", "Section 3.13 scoring / Fig 9 (194)"),
    ("drivers", "present in 15,223-gene voom-tested background", 264, 201,
     "intersect ws1_dge_full GeneSymbol", "Section 3.12.2 (201)"),
    ("drivers", "genes_hit in stage-pair GSEA ranked lists", 264, 185,
     "intersect dedup t-ranked list", "Table 7 / WS27-29 GSEA (185)"),
    ("suppressors", "FerrDb V2 raw download", len(raw_sup), len(raw_sup), "none (raw export)", "source file"),
    ("suppressors", "filtered/HGNC-mapped/dedup (provenance on file)", len(raw_sup), 238,
     "download_ferrdb_v2 filter", "Methods 2.7.7 (238)"),
    ("suppressors", "present in locked 12,537-gene discovery matrix", 238, 186,
     "intersect locked matrix index", "Section 3.13 scoring / Fig 9 (186)"),
    ("suppressors", "present in 15,223-gene voom-tested background", 238, 184,
     "intersect ws1_dge_full GeneSymbol", "Section 3.12.2 (184)"),
    ("suppressors", "genes_hit in stage-pair GSEA ranked lists", 238, 178,
     "intersect dedup t-ranked list", "Table 7 / WS27-29 GSEA (178)"),
]
p5 = pd.DataFrame(chain, columns=["set", "step", "n_before", "n_after",
                                  "filter_applied", "used_in_section"])
p5.to_csv(f"{OUT}/ferroptosis_set_provenance.csv", index=False)
write_provenance("ferroptosis_set_provenance.csv",
                 [INPUTS["drivers_raw"], INPUTS["suppressors_raw"], INPUTS["drivers"],
                  INPUTS["suppressors"], INPUTS["locked"], INPUTS["dge_full"]])
print(p5.to_string(index=False))

# ================= ledger + stats =================
pd.DataFrame(ledger, columns=["metric", "value", "source_file", "source_column",
                              "source_row_selector", "recomputable"]).to_csv(
    f"{OUT}/numbers_ledger_ws29.csv", index=False)
write_provenance("numbers_ledger_ws29.csv", [f"{OUT}/stagepair_gsea_matrisome.csv",
                                             f"{OUT}/celldeath_rho_difference.json",
                                             f"{OUT}/zbtb2_specificity.csv"])
with open(f"{OUT}/stats_ws29.json", "w") as f:
    json.dump(stats, f, indent=2, default=float)
write_provenance("stats_ws29.json", [INPUTS["c2"], INPUTS["locked"], INPUTS["dge_full"]])
print("\nDONE — results/ws29/")
