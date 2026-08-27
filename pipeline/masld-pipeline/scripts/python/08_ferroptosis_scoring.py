#!/usr/bin/env python3
"""
WS26 — Additional ferroptosis analyses (adapted from the attached
ferroptosis_additional_analyses.py, 2026-08-24).

Adaptations vs the attached draft, all forced by data on disk (AGENTS.md rules 1-5):
  1. Expression input is the LOCKED WS15 matrices (Discovery 349 / Fujiwara 213 /
     UCAM 58) from results/ws15/locked_data.pkl — data/expression_matrix.csv is the
     Fujiwara matrix only and carries no provenance, so it is not used.
  2. Gene sets are the filtered FerrDb V2 downloads (264 drivers / 238 suppressors,
     data/ferroptosis_*_ferrdb.csv, provenance: FerrDb V2 browsegene, 2026-08-17).
     The 24-gene constant-copied files named in AGENTS.md section 4 do not exist here.
  3. Analysis 2.3 signature score is DIRECTION-AWARE (z * sign(logFC), discovery
     weights only). The discovery AUROC is in-sample (gene selection and scoring on
     the same 349 patients) and is labelled as such; the publishable numbers are the
     fully external Fujiwara/UCAM AUROCs. Every AUROC gets a bootstrap 95% CI.
  4. Analysis 2.5: AST/ALT/FIB4/NAS/BMI do not exist on disk for the discovery
     cohort — skipped with reason (not invented). Available clinical fields are age
     (all cohorts; within-cohort per AGENTS.md section 6, age is confounded with
     cohort) and sex (corrupted 'fibrosis stage: N' strings for all 216 GSE135251
     rows -> sex analysed only in GSE130970/GSE185051, n stated).
  5. Analyses 2.6 (stage-specific GSEA) and 2.8 (paired-biopsy ferroptosis change)
     are skipped with recorded reasons: no pairwise per-stage DGE on disk; second
     biopsies exist only as raw counts (data/kamzolas/Fujiwara_dataset/), with no
     processed matrix under the locked normalisation protocol.
  6. Seeds: config.yaml no longer exists on disk; SEED=42 used and recorded.

Run:  python3 scripts/python/41_ws26_ferroptosis_additional.py
Outputs -> results/ws26_ferroptosis_additional/
"""

import json
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import kruskal, spearmanr, rankdata, norm
from sklearn.metrics import roc_auc_score
from statsmodels.stats.multitest import multipletests

SEED = 42
N_BOOT = 2000
TOP_N_HEATMAP = 40
OUT = "results/ws26_ferroptosis_additional"
os.makedirs(OUT, exist_ok=True)
MANIFEST = []

FILES = {
    "locked": "results/ws15/locked_data.pkl",
    "meta": "data/discovery_cohort_349.csv",
    "drivers": "data/ferroptosis_driver_ferrdb.csv",
    "suppressors": "data/ferroptosis_suppressor_ferrdb.csv",
    "dge": "results/ws1_signature/ws1_dge_full.csv",
}


def must_exist(path, what):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{what}: {path} not found. Cannot proceed.")
    return path


def log_load(path, rows, used, note):
    MANIFEST.append({"path": path, "rows_in_file": rows,
                     "column_used": used, "note": note})


def md5(path):
    import hashlib
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_provenance(output, inputs, extra=None):
    prov = {
        "output": output,
        "derived_from": [{"path": p, "md5": md5(p), "bytes": os.path.getsize(p)}
                         for p in inputs],
        "script": "scripts/python/41_ws26_ferroptosis_additional.py",
        "git_commit": os.popen("git rev-parse --short HEAD").read().strip(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "random_seed": SEED,
        "n_bootstrap": N_BOOT,
    }
    if extra:
        prov.update(extra)
    with open(f"{OUT}/{output}.provenance.json", "w") as f:
        json.dump(prov, f, indent=2)


def mean_z_score(mat, genes, weights=None):
    """Mean z-score across genes (within-cohort z per gene). weights: +-1 signs."""
    common = [g for g in genes if g in mat.index]
    if not common:
        return None, 0
    sub = mat.loc[common]
    sd = sub.std(axis=1)
    sub = sub[sd > 0]
    if sub.empty:
        return None, 0
    z = ((sub.T - sub.mean(axis=1)) / sd).T
    if weights is not None:
        z = z * np.array([weights[g] for g in sub.index])[:, None]
    return z.mean(axis=0), len(sub)


def dunn_posthoc(df, score_col, group_col="stage"):
    """Dunn's test (no tie correction) with BH across the pair family."""
    ranks = rankdata(df[score_col])
    df = df.assign(_rank=ranks)
    n = len(df)
    out = []
    gs = sorted(df[group_col].unique())
    for i, g1 in enumerate(gs):
        for g2 in gs[i + 1:]:
            a = df[df[group_col] == g1]
            b = df[df[group_col] == g2]
            z = (a._rank.mean() - b._rank.mean()) / np.sqrt(
                (n * (n + 1) / 12) * (1 / len(a) + 1 / len(b)))
            p = 2 * (1 - norm.cdf(abs(z)))
            out.append({"group1": g1, "group2": g2, "n1": len(a), "n2": len(b),
                        "z": z, "p": p})
    res = pd.DataFrame(out)
    res["p_adj"] = multipletests(res["p"], method="fdr_bh")[1]
    return res


def bootstrap_auroc_ci(y, s, seed=SEED, n_boot=N_BOOT):
    y = np.asarray(y)
    s = np.asarray(s)
    rng = np.random.default_rng(seed)
    aucs = []
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    for _ in range(n_boot):
        idx = np.concatenate([rng.choice(pos, len(pos), replace=True),
                              rng.choice(neg, len(neg), replace=True)])
        if len(np.unique(y[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y[idx], s[idx]))
    lo, hi = np.percentile(aucs, [2.5, 97.5])
    return float(lo), float(hi), len(aucs)


def main():
    rng = np.random.default_rng(SEED)
    stats_out = {}

    # ---------------- load ----------------
    for k, p in FILES.items():
        must_exist(p, k)
    with open(FILES["locked"], "rb") as f:
        root = pickle.load(f)
    mats, stages = root["mats"], root["stages"]
    meta = pd.read_csv(FILES["meta"])
    drivers = set(pd.read_csv(FILES["drivers"])["symbol"].dropna().astype(str).str.upper().str.strip())
    suppressors = set(pd.read_csv(FILES["suppressors"])["symbol"].dropna().astype(str).str.upper().str.strip())
    dge = pd.read_csv(FILES["dge"])
    assert len(drivers) == 264 and len(suppressors) == 238, \
        f"FerrDb set sizes wrong: {len(drivers)}/{len(suppressors)} (expected 264/238)"
    assert drivers.isdisjoint(suppressors) is False or True  # overlap allowed in FerrDb; report below
    log_load(FILES["locked"], sum(v.shape[1] for v in mats.values()),
             "mats[Discovery/Fujiwara/UCAM] + stages", "12,537 genes x (349+213+58)")
    log_load(FILES["meta"], len(meta), "sample_id, fibrosis_stage, age, sex, cohort",
             "349 verified-staging discovery samples")
    log_load(FILES["drivers"], len(pd.read_csv(FILES["drivers"])), "symbol", "FerrDb V2 filtered, provenance on file")
    log_load(FILES["suppressors"], len(pd.read_csv(FILES["suppressors"])), "symbol", "FerrDb V2 filtered, provenance on file")
    log_load(FILES["dge"], len(dge), "GeneSymbol, logFC, adj.P.Val", "15,223 voom-tested genes")

    disc = mats["Discovery"]
    assert set(disc.columns) == set(meta.sample_id), "Discovery samples != metadata sample_ids"
    meta = meta.set_index("sample_id").loc[disc.columns]
    stage = stages["Discovery"].loc[disc.columns] if hasattr(stages["Discovery"], "loc") else \
        pd.Series(stages["Discovery"].values, index=disc.columns)
    stage = stage.astype(int)
    assert (stage.values == meta.fibrosis_stage.astype(int).values).all(), \
        "locked stages disagree with discovery_cohort_349.csv fibrosis_stage"
    meta["stage"] = stage

    # ---------------- scores (analysis 2.2) ----------------
    print("[2.2] driver/suppressor mean-z scores, Kruskal-Wallis + Dunn (no tie correction)")
    driver_s, n_dg = mean_z_score(disc, drivers)
    supp_s, n_sg = mean_z_score(disc, suppressors)
    scores = pd.DataFrame({"driver_score": driver_s, "suppressor_score": supp_s,
                           "stage": stage, "cohort": meta.cohort, "age": meta.age})
    scores.index.name = "sample_id"
    scores.to_csv(f"{OUT}/gsva_scores_discovery.csv")
    write_provenance("gsva_scores_discovery.csv",
                     [FILES["locked"], FILES["drivers"], FILES["suppressors"]],
                     extra={"genes_used": {"drivers": n_dg, "suppressors": n_sg},
                            "method": "mean within-sample z-score across gene-set members present in the 12,537-gene locked matrix"})
    stats_out["genes_used"] = {"drivers": n_dg, "suppressors": n_sg}
    stats_out["driver_suppressor_overlap"] = len(drivers & suppressors)

    kw_rows, dunn_frames = [], []
    for col in ["driver_score", "suppressor_score"]:
        groups = [scores[scores.stage == s][col].values for s in range(5)]
        h, p = kruskal(*groups)
        kw_rows.append({"score": col, "KW_H": h, "p": p,
                        "n_per_stage": dict(scores.stage.value_counts().sort_index())})
        d = dunn_posthoc(scores.reset_index(), col)
        d.insert(0, "score", col)
        d.to_csv(f"{OUT}/dunn_{col}.csv", index=False)
        dunn_frames.append(d)
        print(f"    {col}: KW H={h:.3f} p={p:.3e}")
    pd.DataFrame(kw_rows).to_csv(f"{OUT}/kruskal_wallis_results.csv", index=False)
    write_provenance("kruskal_wallis_results.csv", [f"{OUT}/gsva_scores_discovery.csv"])
    for col in ["driver_score", "suppressor_score"]:
        write_provenance(f"dunn_{col}.csv", [f"{OUT}/gsva_scores_discovery.csv"],
                         extra={"note": "Dunn's z without tie correction; BH within score family (10 pairs)"})
    stats_out["kruskal_wallis"] = {r["score"]: {"H": r["KW_H"], "p": r["p"]} for r in kw_rows}
    stats_out["dunn_significant_pairs_fdr05"] = {
        col: int((d.p_adj < 0.05).sum()) for col, d in zip(["driver_score", "suppressor_score"], dunn_frames)}

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, col in zip(axes, ["driver_score", "suppressor_score"]):
        sns.boxplot(data=scores.reset_index(), x="stage", y=col, ax=ax, hue="stage",
                    palette="viridis", order=[0, 1, 2, 3, 4], legend=False)
        ax.set_xlabel("Fibrosis stage")
        ax.set_ylabel("mean z-score")
        ax.set_title(f"{col} (n genes={n_dg if 'driver' in col else n_sg})")
    fig.suptitle(f"Ferroptosis set scores across stages — discovery 349 (drivers {n_dg} / suppressors {n_sg} genes measured)")
    plt.tight_layout()
    fig.savefig(f"{OUT}/fig_2_2_gsva_boxplots.png", dpi=300)
    plt.close(fig)
    write_provenance("fig_2_2_gsva_boxplots.png", [f"{OUT}/gsva_scores_discovery.csv"])

    # ---------------- signature classifier (analysis 2.3) ----------------
    print("[2.3] ferroptosis-DEG signature: discovery (IN-SAMPLE, labelled) + external cohorts")
    dge["gene"] = dge["GeneSymbol"].astype(str).str.upper().str.strip()
    ferro = drivers | suppressors
    fd = dge[dge.gene.isin(ferro) & (dge["adj.P.Val"] < 0.05) & (dge.logFC.abs() > 0.5)]
    sig_genes = fd.gene.tolist()
    weights = dict(zip(fd.gene, np.sign(fd.logFC)))
    pd.DataFrame({"gene": fd.gene, "logFC": fd.logFC, "adj.P.Val": fd["adj.P.Val"],
                  "set": np.where(fd.gene.isin(drivers), "driver", "suppressor")}). \
        to_csv(f"{OUT}/ferroptosis_signature_genes.csv", index=False)
    write_provenance("ferroptosis_signature_genes.csv",
                     [FILES["dge"], FILES["drivers"], FILES["suppressors"]],
                     extra={"selection": "adj.P.Val<0.05 & |logFC|>0.5 & gene in FerrDb filtered sets (against 15,223 tested)"})
    print(f"    ferroptosis DEGs: {len(sig_genes)} "
          f"({sum(g in drivers for g in sig_genes)} driver / {sum(g in suppressors for g in sig_genes)} suppressor)")
    stats_out["signature"] = {"n_degs": len(sig_genes),
                              "n_driver": int(sum(g in drivers for g in sig_genes)),
                              "n_suppressor": int(sum(g in suppressors for g in sig_genes))}

    auroc_rows = []
    for cname, mat, in_sample in [("Discovery", disc, True),
                                  ("Fujiwara", mats["Fujiwara"], False),
                                  ("UCAM", mats["UCAM"], False)]:
        st = stages[cname]
        st = st.loc[mat.columns] if hasattr(st, "loc") else pd.Series(st.values, index=mat.columns)
        st = st.astype(int)
        sc, ng = mean_z_score(mat, sig_genes, weights=weights)
        if sc is None:
            print(f"    {cname}: no signature genes measured — skipped")
            continue
        for label, y in [("F0-2_vs_F3-4", (st >= 3).astype(int)),
                         ("F0-1_vs_F2-4", (st >= 2).astype(int))]:
            yv = y.loc[sc.index].values
            auc = roc_auc_score(yv, sc.values)
            lo, hi, nb = bootstrap_auroc_ci(yv, sc.values)
            auroc_rows.append({"cohort": cname, "threshold": label,
                               "AUROC": auc, "CI95_lo": lo, "CI95_hi": hi,
                               "n": len(yv), "n_positive": int(yv.sum()),
                               "genes_measured": ng, "in_sample_discovery": in_sample,
                               "boot_reps": nb})
            print(f"    {cname:10s} {label:15s} AUROC={auc:.3f} [{lo:.3f},{hi:.3f}] "
                  f"n={len(yv)} pos={int(yv.sum())} genes={ng}"
                  f"{' IN-SAMPLE' if in_sample else ''}")
    auroc_df = pd.DataFrame(auroc_rows)
    auroc_df.to_csv(f"{OUT}/ferroptosis_signature_auroc.csv", index=False)
    write_provenance("ferroptosis_signature_auroc.csv",
                     [FILES["locked"], f"{OUT}/ferroptosis_signature_genes.csv"],
                     extra={"note": "signed mean-z (discovery logFC signs as fixed weights); class-stratified patient bootstrap, seed 42, 2000 reps; discovery row is in-sample (selection+scoring on same patients)"})
    stats_out["auroc"] = {f"{r.cohort}|{r.threshold}":
                          {"auroc": r.AUROC, "ci": [r.CI95_lo, r.CI95_hi],
                           "n": r.n, "n_pos": r.n_positive, "in_sample": bool(r.in_sample_discovery)}
                          for r in auroc_df.itertuples()}

    # ---------------- clinical correlations (analysis 2.5) ----------------
    print("[2.5] available clinical fields: age (all cohorts), sex (GSE130970+GSE185051 only)")
    print("      AST/ALT/FIB4/NAS/BMI: NOT ON DISK for discovery — skipped, not invented")
    rows = []

    def partial_spearman(x, y, covar):
        df = pd.DataFrame({"x": x, "y": y, "c": covar}).dropna()
        rx, ry, rc = (rankdata(df[c]) for c in ("x", "y", "c"))
        res_x = rx - np.polyval(np.polyfit(rc, rx, 1), rc)
        res_y = ry - np.polyval(np.polyfit(rc, ry, 1), rc)
        r, p = stats.pearsonr(res_x, res_y)
        return r, p, len(df)

    age_missing = int(meta["age"].isna().sum())
    age_missing_cohorts = meta[meta["age"].isna()].cohort.value_counts().to_dict()
    print(f"    age missing for {age_missing} rows ({age_missing_cohorts}) — "
          f"pooled age scope spans only cohorts WITH age, incl. paediatric "
          f"GSE185051: cohort-confounded, within-cohort rows are primary")
    stats_out["age_missing"] = {"n": age_missing, "by_cohort": age_missing_cohorts}
    for col in ["driver_score", "suppressor_score"]:
        sub = scores[[col, "age", "stage", "cohort"]].dropna()
        r, p = spearmanr(sub[col], sub["age"])
        rows.append({"score": col, "variable": "age",
                     "scope": "pooled_cohorts_with_age(cohort-confounded)",
                     "rho": r, "p": p, "n": len(sub)})
        pr, pp, pn = partial_spearman(sub[col], sub["age"], sub["stage"])
        rows.append({"score": col, "variable": "age",
                     "scope": "pooled_cohorts_with_age_partial_stage",
                     "rho": pr, "p": pp, "n": pn})
        for coh, gdf in sub.groupby("cohort"):
            r, p = spearmanr(gdf[col], gdf["age"])
            rows.append({"score": col, "variable": "age", "scope": f"within_{coh}",
                         "rho": r, "p": p, "n": len(gdf)})
            pr, pp, pn = partial_spearman(gdf[col], gdf["age"], gdf["stage"])
            rows.append({"score": col, "variable": "age",
                         "scope": f"within_{coh}_partial_stage", "rho": pr, "p": pp, "n": pn})
    sx = meta.copy()
    sx["sex_ok"] = sx.sex.isin(["M", "F"])
    print(f"    sex field corrupted (contains 'fibrosis stage: N') for "
          f"{int((~sx.sex_ok).sum())}/{len(sx)} rows, all GSE135251 — sex analysed in valid cohorts only")
    stats_out["sex_corrupted_rows"] = int((~sx.sex_ok).sum())
    sc2 = scores.join(sx[["sex"]])
    for col in ["driver_score", "suppressor_score"]:
        for coh, gdf in sc2[sc2.sex.isin(["M", "F"])].groupby("cohort"):
            m_ = gdf[gdf.sex == "M"][col]
            f_ = gdf[gdf.sex == "F"][col]
            u, p = stats.mannwhitneyu(m_, f_)
            rows.append({"score": col, "variable": "sex(M vs F)", "scope": f"within_{coh}",
                         "rho": np.nan, "p": p, "n": len(gdf),
                         "note": f"M={len(m_)}, F={len(f_)}, MWU"})
    corr = pd.DataFrame(rows)
    corr["p_adj"] = multipletests(corr.p, method="fdr_bh")[1]
    corr.to_csv(f"{OUT}/ferroptosis_clinical_correlations.csv", index=False)
    write_provenance("ferroptosis_clinical_correlations.csv",
                     [f"{OUT}/gsva_scores_discovery.csv", FILES["meta"]],
                     extra={"note": "Spearman (age), Mann-Whitney (sex); partial = rank-residual method controlling stage; BH across all rows; AST/ALT/FIB4/NAS/BMI absent on disk -> skipped"})
    stats_out["age_pooled"] = {r.score: {"rho": r.rho, "p": r.p}
                               for r in corr.itertuples() if r.variable == "age" and r.scope == "pooled"}

    # ---------------- heatmap (analysis 2.1) ----------------
    print("[2.1] heatmap of top-40 variable ferroptosis genes")
    present = [g for g in ferro if g in disc.index]
    var = disc.loc[present].var(axis=1).sort_values(ascending=False)
    top = var.head(TOP_N_HEATMAP).index
    z = ((disc.loc[top].T - disc.loc[top].mean(axis=1)) / disc.loc[top].std(axis=1)).T
    order = stage.sort_values().index
    z = z[order]
    fig, ax = plt.subplots(figsize=(14, 9))
    sns.heatmap(z, cmap="RdBu_r", center=0, ax=ax, cbar_kws={"label": "z (within gene)"})
    ax.set_title(f"Top {TOP_N_HEATMAP} ferroptosis genes by variance — discovery 349, samples ordered by stage")
    ax.set_xlabel("samples (sorted by stage)")
    plt.tight_layout()
    fig.savefig(f"{OUT}/fig_2_1_ferroptosis_heatmap.png", dpi=300)
    plt.close(fig)
    write_provenance("fig_2_1_ferroptosis_heatmap.png",
                     [FILES["locked"], FILES["drivers"], FILES["suppressors"]])

    # ---------------- skipped analyses ----------------
    stats_out["skipped"] = {
        "2.6_stage_specific_gsea": "requires pairwise per-stage DGE; only the F3-4 vs F0-2 contrast exists on disk (ws1_dge_full.csv)",
        "2.8_paired_biopsy_ferroptosis": "second-biopsy expression exists only as raw counts (data/kamzolas/Fujiwara_dataset/raw_counts_fujiwara.csv); no processed paired matrix under the locked normalisation — computing scores would require a new normalisation path, not run",
        "2.5_AST_ALT_FIB4_NAS_BMI": "not present in any discovery metadata file on disk",
    }

    # ---------------- manifest + stats ----------------
    pd.DataFrame(MANIFEST).to_csv(f"{OUT}/load_manifest.csv", index=False)
    write_provenance("load_manifest.csv", list(FILES.values()))
    stats_out["analysis_date"] = datetime.now(timezone.utc).isoformat()
    stats_out["seed"] = SEED
    with open(f"{OUT}/stats_ws26.json", "w") as f:
        json.dump(stats_out, f, indent=2)
    write_provenance("stats_ws26.json", list(FILES.values()))
    print("\nDONE — outputs in", OUT)


if __name__ == "__main__":
    main()
