#!/usr/bin/env python3
"""
16_fujiwara_validation.py

External and longitudinal cohort validation on the Fujiwara/Hoshida cohort
(GSE192959 / GSE193066 / GSE193080). Computes ferroptosis driver/suppressor
coupling, stage-stratified gradients, age confounder diagnostics, and
longitudinal paired-biopsy progression statistics.

Usage:
    python 16_fujiwara_validation.py \\
        --expr data/expression_matrix.csv \\
        --meta data/metadata_with_ferroptosis_scores.csv \\
        --drivers data/ferroptosis_drivers.csv \\
        --suppressors data/ferroptosis_suppressors.csv \\
        --out results/fujiwara
"""

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

DEFAULT_DRIVERS = [
    "ACSL4", "LPCAT3", "ALOX15", "ALOX5", "TFRC", "NCOA4", "PTGS2", "SAT1",
    "TP53", "BAX", "IL1B", "TIMP1", "GJA1", "CGAS", "STING1", "EGR1", "RRM2",
    "TGFB1", "CYBB", "POR", "ATP5MC3", "BID", "KDM5A", "CFL1"
]

DEFAULT_SUPPRESSORS = [
    "GPX4", "SLC7A11", "FTH1", "FTL", "NFE2L2", "HMOX1", "HSPA5",
    "CDKN1A", "TXN", "TMSB4X", "ENPP2", "VDR", "MGST1", "CISD1", "CISD2",
    "GCLC", "GCLM", "AIFM2", "CHMP5", "COPZ1", "TMBIM4", "ISCU", "GABARAPL2",
    "SLC3A2"
]


def load(expr_p, meta_p):
    E = pd.read_csv(expr_p, index_col=0, low_memory=False)
    m = pd.read_csv(meta_p, low_memory=False)
    m["age"] = pd.to_numeric(m["age:ch1"], errors="coerce")
    mm = m.set_index(["dataset", "title"])
    rows = []
    for c in E.columns:
        d, t = c.split(".", 1)
        if (d, t) in mm.index:
            r = mm.loc[(d, t)].copy()
            r["col"] = c
            r["ds"] = d
            r["ttl"] = t
            rows.append(r)
    M = pd.DataFrame(rows).reset_index(drop=True)
    M = M.loc[:, ~M.columns.duplicated()]
    return E[M.col.tolist()], M, m


def score(E, M, drivers, suppressors):
    z = E.sub(E.mean(axis=1), axis=0).div(E.std(axis=1).replace(0, np.nan), axis=0)
    D = [g for g in drivers if g in E.index]
    S = [g for g in suppressors if g in E.index]
    M["driver"] = z.loc[D].mean(axis=0).values
    M["suppressor"] = z.loc[S].mean(axis=0).values
    M["stage"] = pd.to_numeric(M.fibrosis_stage)
    return z, D, S


def permutation_null(z, E, nD, nS, obs, n=1000, seed=42):
    """Null distribution of coupling between size-matched random gene sets."""
    rng = np.random.default_rng(seed)
    pool = [g for g in E.index if E.loc[g].std() > 0]
    null = np.array([
        np.corrcoef(
            z.loc[rng.choice(pool, nD, replace=False)].mean(axis=0),
            z.loc[rng.choice(pool, nS, replace=False)].mean(axis=0)
        )[0, 1]
        for _ in range(n)
    ])
    return {
        "observed": float(obs),
        "null_mean": float(null.mean()),
        "null_sd": float(null.std()),
        "null_p95": float(np.percentile(null, 95)),
        "empirical_p": float((null >= obs).mean()),
        "excess_over_null": float(obs - null.mean())
    }


def paired_table(m):
    """Build the 58-patient paired-biopsy progression table."""
    g = m[m.dataset == "GSE193066"].copy()
    g["stem"] = g.title.str.extract(r"(\d+)")[0]
    b1 = g[g["biopsy:ch1"] == "1st biopsy"].drop_duplicates("stem").set_index("stem")
    b2 = g[g["biopsy:ch1"] == "2nd biopsy"].drop_duplicates("stem").set_index("stem")
    idx = sorted(set(b1.index) & set(b2.index))
    P = pd.DataFrame({
        "title_b1": b1.loc[idx, "title"],
        "F_b1": b1.loc[idx, "fibrosis_stage"],
        "age_b1": b1.loc[idx, "age"],
        "F_b2": b2.loc[idx, "fibrosis_stage"],
        "age_b2": b2.loc[idx, "age"]
    })
    P["interval_yr"] = P.age_b2 - P.age_b1
    P["deltaF"] = P.F_b2 - P.F_b1
    P["progressed"] = (P.deltaF >= 1).astype(int)
    return P


def auc_ci(y, s, seed=0, n=2000):
    y, s = np.asarray(y), np.asarray(s, dtype=float)
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) == 2:
            bs.append(roc_auc_score(y[i], s[i]))
    return float(roc_auc_score(y, s)), [float(x) for x in np.percentile(bs, [2.5, 97.5])]


def main():
    ap = argparse.ArgumentParser(description="Fujiwara Cohort Validation")
    ap.add_argument("--expr", required=True, help="Path to expression matrix CSV")
    ap.add_argument("--meta", required=True, help="Path to metadata CSV")
    ap.add_argument("--drivers", help="Path to ferroptosis drivers CSV")
    ap.add_argument("--suppressors", help="Path to ferroptosis suppressors CSV")
    ap.add_argument("--out", default="results/fujiwara", help="Output directory")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    drivers = (
        pd.read_csv(a.drivers).iloc[:, 0].astype(str).tolist()
        if a.drivers else DEFAULT_DRIVERS
    )
    suppressors = (
        pd.read_csv(a.suppressors).iloc[:, 0].astype(str).tolist()
        if a.suppressors else DEFAULT_SUPPRESSORS
    )

    E, M, m = load(a.expr, a.meta)
    z, D, S = score(E, M, drivers, suppressors)
    R = {"n_samples": int(len(M)), "n_driver_genes": len(D), "n_suppressor_genes": len(S)}

    # 1. Coupling against null
    r, p = stats.pearsonr(M.driver, M.suppressor)
    R["coupling"] = {"r": float(r), "p": float(p)}
    R["coupling_null"] = permutation_null(z, E, len(D), len(S), r)

    # 2. Stage gradient
    R["stage_gradient"] = {
        k: dict(zip(("rho", "p"), map(float, stats.spearmanr(M[k], M.stage))))
        for k in ("driver", "suppressor")
    }

    # 3. Age effect within each adult cohort
    R["age_within_cohort"] = {}
    for d, gg in M.groupby("ds"):
        R["age_within_cohort"][d] = {
            "n": int(len(gg)),
            "age_min": float(gg.age.min()),
            "age_max": float(gg.age.max()),
            "driver_r": float(stats.pearsonr(gg.driver, gg.age)[0]),
            "suppressor_r": float(stats.pearsonr(gg.suppressor, gg.age)[0])
        }

    # 4. TIMP1-TXN within cohort
    if "TIMP1" in E.index and "TXN" in E.index:
        t1, tx = E.loc["TIMP1"].values, E.loc["TXN"].values
        R["timp1_txn"] = {
            "pooled_r": float(stats.pearsonr(t1, tx)[0]),
            "within": {
                d: float(stats.pearsonr(t1[gg.index], tx[gg.index])[0])
                for d, gg in M.groupby("ds")
            }
        }

    # 5. Longitudinal paired biopsy progression
    P = paired_table(m)
    M["stem"] = pd.Series(M.ttl.values).str.extract(r"(\d+)")[0].values
    b1 = M[M.ds == "GSE193066"].drop_duplicates("stem").set_index("stem")
    P2 = P.join(b1[["driver", "suppressor"]], how="inner")
    y = P2.progressed.values
    R["longitudinal"] = {
        "n_paired": int(len(P)),
        "n_with_baseline_expr": int(len(P2)),
        "n_progressed": int(P2.progressed.sum()),
        "n_regressed": int((P.deltaF <= -1).sum()),
        "median_interval_yr": float(P.interval_yr.median()),
        "negative_intervals": int((P.interval_yr < 0).sum()),
        "auroc": {},
        "by_baseline_stage": {}
    }
    for nm, v in [
        ("driver", P2.driver.values),
        ("suppressor", P2.suppressor.values),
        ("poised_mean", ((P2.driver + P2.suppressor) / 2).values),
        ("baseline_stage_neg", -P2.F_b1.values.astype(float))
    ]:
        auc_val, ci = auc_ci(y, v)
        R["longitudinal"]["auroc"][nm] = {"auroc": auc_val, "ci95": ci}
    for f, g in P2.groupby("F_b1"):
        R["longitudinal"]["by_baseline_stage"][f"F{f}"] = {
            "n": int(len(g)),
            "n_progressed": int(g.progressed.sum())
        }

    (out / "fujiwara_validation.json").write_text(json.dumps(R, indent=2))
    P2.to_csv(out / "paired_progression.csv")
    print(json.dumps(R, indent=2))


if __name__ == "__main__":
    main()
