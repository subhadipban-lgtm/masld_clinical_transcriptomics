#!/usr/bin/env python3
"""
19_kamzolas_integration.py

Uses the Kamzolas et al. (Nat Metab 2026) public data release:
  A. REAL paired-biopsy molecular longitudinal analysis (n=58 paired biopsies).
  B. Independent external validation across Fujiwara, EPoS, and UCAM/Sanyal.
  C. Gene-gene regulatory layer from CollecTRI network for KG enrichment.
"""

import argparse
import datetime
import json
import math
import os
from pathlib import Path
import subprocess
import zipfile

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import roc_auc_score


def get_git_commit(kam_dir: Path) -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(kam_dir.parent)).decode().strip()
    except Exception:
        try:
            return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
        except Exception:
            return "unknown"


def write_provenance(out_file: Path, derived_from: list, script_cmd: str, git_commit: str):
    prov = {
        "output": str(out_file),
        "derived_from": [str(d) for d in derived_from],
        "script": script_cmd,
        "git_commit": git_commit,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat()
    }
    prov_file = out_file.parent / (out_file.name + ".provenance.json")
    with open(prov_file, "w") as f:
        json.dump(prov, f, indent=2)


def load_gene_sets(drivers_p, suppressors_p):
    D = pd.read_csv(drivers_p).iloc[:, 0].astype(str).str.strip().tolist()
    S = pd.read_csv(suppressors_p).iloc[:, 0].astype(str).str.strip().tolist()
    if len(D) < 50 or len(S) < 50:
        raise ValueError(f"ERROR: {len(D)} drivers / {len(S)} suppressors. Below the 50-gene threshold.")
    return D, S


def quantile_normalize(df: pd.DataFrame) -> pd.DataFrame:
    rank_mean = df.stack().groupby(df.rank(method="first").stack().astype(int)).mean()
    return df.rank(method="min").stack().astype(int).map(rank_mean).unstack()


def ferroptosis_scores(cpm_log: pd.DataFrame, driver_ids: list, suppressor_ids: list):
    z = cpm_log.sub(cpm_log.mean(axis=1), axis=0).div(cpm_log.std(axis=1).replace(0, np.nan), axis=0)
    D = [g for g in driver_ids if g in z.index]
    S = [g for g in suppressor_ids if g in z.index]
    out = pd.DataFrame({"driver": z.loc[D].mean(axis=0), "suppressor": z.loc[S].mean(axis=0)})
    out["axis"] = out.driver - out.suppressor
    out["poised"] = (out.driver + out.suppressor) / 2
    return out, len(D), len(S)


def loocv_auroc(df: pd.DataFrame, cols: list, y: np.ndarray) -> float:
    X = df[cols].astype(float).values
    pred = np.zeros(len(y))
    for tr, te in LeaveOneOut().split(X):
        m = LogisticRegression(penalty=None, max_iter=2000).fit(X[tr], y[tr])
        pred[te] = m.predict_proba(X[te])[:, 1]
    return float(roc_auc_score(y, pred))


def meta_r(rs: list, ns: list):
    zs = np.arctanh(rs)
    vars_ = 1.0 / (np.array(ns) - 3)
    weights = 1.0 / vars_
    z_fixed = np.sum(weights * zs) / np.sum(weights)
    Q = float(np.sum(weights * (zs - z_fixed)**2))
    k = len(rs)
    df = k - 1
    c = np.sum(weights) - np.sum(weights**2) / np.sum(weights)
    tau2 = float(max(0.0, (Q - df) / c))
    re_weights = 1.0 / (vars_ + tau2)
    z_re = np.sum(re_weights * zs) / np.sum(re_weights)
    se_re = 1.0 / np.sqrt(np.sum(re_weights))
    ci_low = float(np.tanh(z_re - 1.96 * se_re))
    ci_high = float(np.tanh(z_re + 1.96 * se_re))
    r_re = float(np.tanh(z_re))
    z_score = z_re / se_re
    p_val = float(2 * (1 - 0.5 * (1 + math.erf(abs(z_score) / np.sqrt(2)))))
    return {"r_meta": r_re, "ci_95": [ci_low, ci_high], "tau2": tau2, "Q": Q, "p_val": p_val}


# ------------------------------------------------------- A. paired analysis

def analysis_paired(kam: Path, drivers_p: Path, suppressors_p: Path, out: Path):
    C = pd.read_csv(kam / "Fujiwara_dataset/raw_counts_fujiwara.csv", index_col=0)
    cpm = np.log2(C / C.sum() * 1e6 + 1)
    cpm = cpm[cpm.std(axis=1) > 0]

    D_sym, S_sym = load_gene_sets(drivers_p, suppressors_p)
    ens_map = pd.read_csv(kam / "ensg_map.csv")
    d_ids = [e for s, e in zip(ens_map.symbol, ens_map.ensembl) if s in set(D_sym) and e in cpm.index]
    s_ids = [e for s, e in zip(ens_map.symbol, ens_map.ensembl) if s in set(S_sym) and e in cpm.index]

    # Technical covariates & Housekeeping genes
    hk_symbols = ["ACTB", "GAPDH", "B2M", "RPL13A", "HPRT1", "PPIA", "SDHA", "TBP", "UBC", "YWHAZ"]
    ens_tab = pd.read_csv(kam / "ensembl_mapping.tsv", sep="\t")
    ens_dict = dict(zip(ens_tab["external_gene_name"].astype(str).str.strip(), ens_tab["ensembl_gene_id"].astype(str).str.strip()))
    hk_ids = [ens_dict[g] for g in hk_symbols if g in ens_dict and ens_dict[g] in cpm.index]

    z = cpm.sub(cpm.mean(axis=1), axis=0).div(cpm.std(axis=1).replace(0, np.nan), axis=0)
    raw_driver = z.loc[d_ids].mean(axis=0)
    raw_supp = z.loc[s_ids].mean(axis=0)
    hk_mean = z.loc[hk_ids].mean(axis=0)
    lib_size = np.log10(C.sum())
    genes_det = (C > 0).sum()

    tech_X = pd.DataFrame({
        "log_libsize": lib_size,
        "genes_detected": genes_det,
        "hk_mean": hk_mean
    }).loc[cpm.columns]

    adj_driver = raw_driver - LinearRegression().fit(tech_X, raw_driver).predict(tech_X)
    adj_supp = raw_supp - LinearRegression().fit(tech_X, raw_supp).predict(tech_X)

    sc = pd.DataFrame({
        "driver": raw_driver,
        "suppressor": raw_supp,
        "axis": raw_driver - raw_supp,
        "poised": (raw_driver + raw_supp) / 2,
        "driver_adj": adj_driver,
        "suppressor_adj": adj_supp,
        "axis_adj": adj_driver - adj_supp,
    })
    sc["patient"] = [i.rsplit("_", 1)[0] for i in sc.index]
    sc["biopsy"] = [i.rsplit("_", 1)[1] for i in sc.index]
    W = sc.pivot(index="patient", columns="biopsy").dropna()
    W.columns = [f"{a}_{b}" for a, b in W.columns]

    xl = pd.ExcelFile(kam / "Fujiwara_dataset/Both biopsies - refined dataset.xlsx")
    b1 = xl.parse("1st_biopsy").set_index("Patient")
    b2 = xl.parse("2nd_biopsy").set_index("Patient")
    P = (W.join(b1[["Histology.fibrosis", "Histology.NAS", "Age", "BMI", "Diabetes",
                    "Date_Tissue.acquisition", "PNPLA3_rs738409"]])
           .join(b2[["Histology.fibrosis_2", "NAS_2", "Date_2nd_Bx"]])
           .dropna(subset=["Histology.fibrosis", "Histology.fibrosis_2"]))

    # Date-derived interval
    P["interval_yr"] = (pd.to_datetime(P["Date_2nd_Bx"]) - pd.to_datetime(P["Date_Tissue.acquisition"])).dt.days / 365.25
    P["dF"] = P["Histology.fibrosis_2"] - P["Histology.fibrosis"]
    P["dNAS"] = P["NAS_2"] - P["Histology.NAS"]
    P["d_axis"] = P.axis_2 - P.axis_1
    P["d_poised"] = P.poised_2 - P.poised_1
    P["progressed"] = (P.dF >= 1).astype(int)
    y = P.progressed.values

    # Prognostic LOOCV AUROC
    auc_stage = loocv_auroc(P, ["Histology.fibrosis"], y)
    auc_axis = loocv_auroc(P, ["axis_1"], y)
    auc_combo = loocv_auroc(P, ["Histology.fibrosis", "axis_1"], y)
    auc_stage_nas = loocv_auroc(P, ["Histology.fibrosis", "Histology.NAS"], y)
    delta_auc_obs = auc_combo - auc_stage

    # Permutation test for delta AUROC
    np.random.seed(42)
    n_perm = 1000
    perm_deltas = []
    df_perm = P.copy()
    for _ in range(n_perm):
        df_perm["axis_perm"] = np.random.permutation(P["axis_1"].values)
        auc_p = loocv_auroc(df_perm, ["Histology.fibrosis", "axis_perm"], y)
        perm_deltas.append(auc_p - auc_stage)
    perm_deltas = np.array(perm_deltas)
    emp_p = float((perm_deltas >= delta_auc_obs).mean())

    # Shifts with and without adjustment
    shift_res = {}
    for tag in ["raw", "adj"]:
        shift_res[tag] = {}
        for feat in ["driver", "suppressor", "axis"]:
            col_b1 = f"{feat}_{tag}_1" if tag == "adj" else f"{feat}_1"
            col_b2 = f"{feat}_{tag}_2" if tag == "adj" else f"{feat}_2"
            b1_v = P[col_b1]
            b2_v = P[col_b2]
            w_stat, w_p = stats.wilcoxon(b1_v, b2_v)
            t_stat, t_p = stats.ttest_rel(b2_v, b1_v)
            shift_res[tag][feat] = {
                "median_b1": float(b1_v.median()),
                "median_b2": float(b2_v.median()),
                "mean_diff": float(np.mean(b2_v - b1_v)),
                "wilcoxon_p": float(w_p),
                "paired_t_p": float(t_p)
            }

    R = {
        "n_paired": int(len(P)),
        "n_driver_genes": len(d_ids),
        "n_suppressor_genes": len(s_ids),
        "median_interval_yr": float(P.interval_yr.median()),
        "n_progressed": int(y.sum()),
        "n_regressed": int((P.dF <= -1).sum()),
        "n_stable": int((P.dF == 0).sum()),
        "delta_vs_delta": {
            "d_axis_vs_dF": {"rho": float(stats.spearmanr(P.d_axis, P.dF)[0]), "p": float(stats.spearmanr(P.d_axis, P.dF)[1])},
            "d_axis_vs_dNAS": {"rho": float(stats.spearmanr(P.d_axis, P.dNAS)[0]), "p": float(stats.spearmanr(P.d_axis, P.dNAS)[1])},
            "d_poised_vs_dF": {"rho": float(stats.spearmanr(P.d_poised, P.dF)[0]), "p": float(stats.spearmanr(P.d_poised, P.dF)[1])},
            "d_poised_vs_dNAS": {"rho": float(stats.spearmanr(P.d_poised, P.dNAS)[0]), "p": float(stats.spearmanr(P.d_poised, P.dNAS)[1])},
        },
        "within_patient_shift": shift_res,
        "prognostic_loocv_auroc": {
            "stage_alone": auc_stage,
            "axis_alone": auc_axis,
            "stage_plus_axis": auc_combo,
            "stage_plus_NAS": auc_stage_nas,
            "incremental_delta_auroc": delta_auc_obs,
            "incremental_permutation_p": emp_p,
            "n_permutations": n_perm
        }
    }

    out.mkdir(parents=True, exist_ok=True)
    out_csv = out / "fujiwara_real_paired.csv"
    out_json = out / "stats_paired.json"
    P.to_csv(out_csv)
    out_json.write_text(json.dumps(R, indent=2))

    # Also copy to root results if needed
    root_out_csv = Path("fujiwara_real_paired.csv")
    P.to_csv(root_out_csv)

    git_commit = get_git_commit(kam)
    derived_from = [
        kam / "Fujiwara_dataset/raw_counts_fujiwara.csv",
        kam / "Fujiwara_dataset/Both biopsies - refined dataset.xlsx",
        drivers_p,
        suppressors_p,
        kam / "ensg_map.csv"
    ]
    write_provenance(out_csv, derived_from, "19_kamzolas_integration.py --analysis paired", git_commit)
    write_provenance(out_json, derived_from, "19_kamzolas_integration.py --analysis paired", git_commit)
    write_provenance(root_out_csv, derived_from, "19_kamzolas_integration.py --analysis paired", git_commit)

    print("Paired analysis complete. JSON summary written to", out_json)
    return R


# ------------------------------------------------- B. external validation

def analyze_single_cohort(name: str, expr_df: pd.DataFrame, meta_df: pd.DataFrame, stage_col: str, drivers: list, suppressors: list, ens_dict: dict, n_null=1000):
    is_ens = any(str(idx).startswith("ENSG") for idx in expr_df.index[:100])
    if is_ens:
        d_ids = [ens_dict[s] for s in drivers if s in ens_dict and ens_dict[s] in expr_df.index]
        s_ids = [ens_dict[s] for s in suppressors if s in ens_dict and ens_dict[s] in expr_df.index]
        timp1_id = ens_dict.get("TIMP1")
        txn_id = ens_dict.get("TXN")
    else:
        d_ids = [s for s in drivers if s in expr_df.index]
        s_ids = [s for s in suppressors if s in expr_df.index]
        timp1_id = "TIMP1" if "TIMP1" in expr_df.index else None
        txn_id = "TXN" if "TXN" in expr_df.index else None

    z = expr_df.sub(expr_df.mean(axis=1), axis=0).div(expr_df.std(axis=1).replace(0, np.nan), axis=0)
    d_score = z.loc[d_ids].mean(axis=0)
    s_score = z.loc[s_ids].mean(axis=0)
    axis_score = d_score - s_score

    r_obs, p_obs = stats.pearsonr(d_score, s_score)

    all_genes = expr_df.index.tolist()
    nD, nS = len(d_ids), len(s_ids)
    np.random.seed(42)
    null_rs = []
    for _ in range(n_null):
        perm_genes = np.random.choice(all_genes, size=nD + nS, replace=False)
        d_null = z.loc[perm_genes[:nD]].mean(axis=0)
        s_null = z.loc[perm_genes[nD:]].mean(axis=0)
        r_null, _ = stats.pearsonr(d_null, s_null)
        null_rs.append(r_null)
    null_rs = np.array(null_rs)
    null_mean = float(np.mean(null_rs))
    null_sd = float(np.std(null_rs))
    emp_p = float((null_rs >= r_obs).mean())
    excess = float(r_obs - null_mean)

    common_meta = meta_df.loc[expr_df.columns].dropna(subset=[stage_col])
    stages = pd.to_numeric(common_meta[stage_col], errors="coerce")
    stages = stages[stages.notna()]
    stage_axis = axis_score.loc[stages.index]
    stage_d = d_score.loc[stages.index]
    stage_s = s_score.loc[stages.index]
    rho_axis, p_axis = stats.spearmanr(stages, stage_axis)
    rho_d, p_d = stats.spearmanr(stages, stage_d)
    rho_s, p_s = stats.spearmanr(stages, stage_s)

    timp1_txn_r, timp1_txn_p = np.nan, np.nan
    if timp1_id and txn_id and timp1_id in expr_df.index and txn_id in expr_df.index:
        timp1_txn_r, timp1_txn_p = stats.pearsonr(expr_df.loc[timp1_id], expr_df.loc[txn_id])

    return {
        "cohort": name,
        "n": int(expr_df.shape[1]),
        "n_stage": int(len(stages)),
        "nD": nD,
        "nS": nS,
        "coupling_r": float(r_obs),
        "coupling_p": float(p_obs),
        "null_mean": null_mean,
        "null_sd": null_sd,
        "excess_over_null": excess,
        "coupling_emp_p": emp_p,
        "stage_rho_axis": float(rho_axis),
        "stage_p_axis": float(p_axis),
        "stage_rho_driver": float(rho_d),
        "stage_p_driver": float(p_d),
        "stage_rho_suppressor": float(rho_s),
        "stage_p_suppressor": float(p_s),
        "timp1_txn_r": float(timp1_txn_r),
        "timp1_txn_p": float(timp1_txn_p)
    }


def analysis_external(kam: Path, drivers_p: Path, suppressors_p: Path, out: Path):
    D_sym, S_sym = load_gene_sets(drivers_p, suppressors_p)
    ens_tab = pd.read_csv(kam / "ensembl_mapping.tsv", sep="\t")
    ens_dict = dict(zip(ens_tab["external_gene_name"].astype(str).str.strip(), ens_tab["ensembl_gene_id"].astype(str).str.strip()))

    # 1. Fujiwara cohort (n=213)
    # Check if expression_matrix.csv exists in data/ or masld-cdss/data/
    expr_p = Path("masld-cdss/data/expression_matrix.csv") if Path("masld-cdss/data/expression_matrix.csv").exists() else Path("data/expression_matrix.csv")
    meta_p = Path("masld-cdss/data/metadata_with_ferroptosis_scores.csv") if Path("masld-cdss/data/metadata_with_ferroptosis_scores.csv").exists() else Path("data/metadata_with_ferroptosis_scores.csv")
    fuj_expr = pd.read_csv(expr_p, index_col=0)
    fuj_meta = pd.read_csv(meta_p)
    fuj_meta["col"] = fuj_meta["dataset"] + "." + fuj_meta["title"]
    fuj_meta = fuj_meta.set_index("col")
    fuj_meta["fibrosis_stage"] = pd.to_numeric(fuj_meta["fibrosis stage:ch1"], errors="coerce")
    res_fuj = analyze_single_cohort("Fujiwara", fuj_expr, fuj_meta, "fibrosis_stage", D_sym, S_sym, ens_dict)

    # 2. EPoS cohort (n=168)
    with zipfile.ZipFile(kam / "EPoS_dataset/EPoS_counts.tsv.zip") as z:
        with z.open("EPoS_counts.tsv") as f:
            epos_raw = pd.read_csv(f, sep="\t", index_col=0)
    epos_raw.columns = epos_raw.columns.str.strip()
    epos_meta = pd.read_csv(kam / "EPoS_dataset/epos_metadata.csv").set_index("GEO_ID")
    common_epos = [c for c in epos_raw.columns if c in epos_meta.index]
    epos_raw = epos_raw[common_epos]
    epos_qn = quantile_normalize(np.log2(epos_raw + 1))
    res_epos = analyze_single_cohort("EPoS", epos_qn, epos_meta, "Fibrosis.stage", D_sym, S_sym, ens_dict)

    # 3. UCAM/Sanyal cohort (n=135)
    ucam_raw = pd.read_csv(kam / "ucam_sanyal/counts_matrix.csv", index_col=0)
    ucam_meta = pd.read_csv(kam / "ucam_sanyal/metadata.csv").set_index("Sample name")
    common_ucam = [c for c in ucam_raw.columns if c in ucam_meta.index]
    ucam_raw = ucam_raw[common_ucam]
    ucam_cpm = np.log2(ucam_raw / ucam_raw.sum() * 1e6 + 1)
    res_ucam = analyze_single_cohort("UCAM/Sanyal", ucam_cpm, ucam_meta, "Fibrosis", D_sym, S_sym, ens_dict)

    cohorts_res = [res_fuj, res_epos, res_ucam]
    ns = [r["n"] for r in cohorts_res]
    coupling_rs = [r["coupling_r"] for r in cohorts_res]
    stage_rhos = [r["stage_rho_axis"] for r in cohorts_res]

    meta_coupling = meta_r(coupling_rs, ns)
    meta_stage = meta_r(stage_rhos, ns)

    R_ext = {
        "cohorts": cohorts_res,
        "meta_analysis": {
            "coupling_meta": meta_coupling,
            "stage_gradient_meta": meta_stage
        }
    }

    out.mkdir(parents=True, exist_ok=True)
    out_json = out / "stats_external.json"
    out_json.write_text(json.dumps(R_ext, indent=2))

    git_commit = get_git_commit(kam)
    derived_from = [
        expr_p,
        meta_p,
        kam / "EPoS_dataset/EPoS_counts.tsv.zip",
        kam / "EPoS_dataset/epos_metadata.csv",
        kam / "ucam_sanyal/counts_matrix.csv",
        kam / "ucam_sanyal/metadata.csv",
        drivers_p,
        suppressors_p
    ]
    write_provenance(out_json, derived_from, "19_kamzolas_integration.py --analysis external", git_commit)
    print("External cohort validation complete. Summary written to", out_json)
    return R_ext


# ---------------------------------------------------------- D. KG enrichment

def analysis_kg_edges(kam: Path, out: Path):
    tri = pd.read_csv(kam / "collectTRI_network.tsv", sep="\t")
    tri.columns = [c.strip('"') for c in tri.columns]
    for c in ("tf", "target", "confidence"):
        tri[c] = tri[c].astype(str).str.strip('"')
    edges = tri.rename(columns={"tf": "source", "target": "target"})
    edges["relation"] = np.where(edges.mor.astype(float) > 0, "activates", "represses")
    edges["evidence"] = "CollecTRI"

    out.mkdir(parents=True, exist_ok=True)
    out_csv = out / "kg_gene_gene_edges_collectri.csv"
    out_json = out / "stats_kg_edges.json"
    edges[["source", "target", "relation", "confidence", "evidence"]].to_csv(out_csv, index=False)

    stats_ = {
        "collectri_edges": int(len(edges)),
        "unique_tfs": int(edges.source.nunique()),
        "unique_targets": int(edges.target.nunique()),
        "high_confidence_A": int((edges.confidence == "A").sum())
    }
    out_json.write_text(json.dumps(stats_, indent=2))

    git_commit = get_git_commit(kam)
    write_provenance(out_csv, [kam / "collectTRI_network.tsv"], "19_kamzolas_integration.py --analysis kg-edges", git_commit)
    write_provenance(out_json, [kam / "collectTRI_network.tsv"], "19_kamzolas_integration.py --analysis kg-edges", git_commit)
    print("KG edge extraction complete. Summary written to", out_json)
    return stats_


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--kam-dir", default=Path("data/kamzolas"), type=Path)
    ap.add_argument("--drivers", default=Path("data/ferroptosis_driver_ferrdb.csv"), type=Path)
    ap.add_argument("--suppressors", default=Path("data/ferroptosis_suppressor_ferrdb.csv"), type=Path)
    ap.add_argument("--analysis", required=True, choices=["paired", "external", "kg-edges"])
    ap.add_argument("--out", default=Path("results/kamzolas"), type=Path)
    a = ap.parse_args()

    if a.analysis == "paired":
        analysis_paired(a.kam_dir, a.drivers, a.suppressors, a.out)
    elif a.analysis == "external":
        analysis_external(a.kam_dir, a.drivers, a.suppressors, a.out)
    else:
        analysis_kg_edges(a.kam_dir, a.out)
