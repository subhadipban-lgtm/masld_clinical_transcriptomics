#!/usr/bin/env python3
"""
WS30c — (1) random-set null and (2) technical-covariate adjustment for the Fujiwara
ferroptosis-score inversion (user-requested, 2026-08-25).

PRE-COMMITTED RULES (user-stated, before computing):
  N1: if the null distribution of rho (1,000 random gene sets of the same size, identical
      mean-z estimator, same cohort) is centred near zero and the observed -0.224 sits in
      its tail, the ferroptosis result is SPECIFIC. If random sets also land near -0.2,
      the finding is a GLOBAL MEAN-Z DRIFT artifact.
  N2: if stage association survives adjustment for technical covariates, the inversion is
      ROBUST; if it attenuates toward zero, it is a technical artifact.

Data constraint discovered and disclosed: the locked Fujiwara matrix is post-normalisation
GEO log2 expression with NO zero entries — genes-detected is degenerate (12,537/12,537 for
every sample) and true library size is not recoverable for the 213 cross-sectional samples.
Covariates therefore: (a) recoverable proxies on all 213 — global all-gene mean-z (the
drift itself), per-sample linear-space expression sum (library-size-like), per-sample mean
log2 expression; (b) the real thing (raw-count library size and genes detected) on the
58-patient baseline subset from data/kamzolas/Fujiwara_dataset/raw_counts_fujiwara.csv.
"""
import json
import os
import pickle
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, rankdata

SEED = 42
N_NULL = 1000
OUT = "results/ws30"
print("PRE-COMMITTED RULES N1/N2 — stated in docstring before computing.")

with open("results/ws15/locked_data.pkl", "rb") as f:
    root = pickle.load(f)
drivers = set(pd.read_csv("data/ferroptosis_driver_ferrdb.csv")["symbol"].str.upper().str.strip())
suppressors = set(pd.read_csv("data/ferroptosis_suppressor_ferrdb.csv")["symbol"].str.upper().str.strip())

rng = np.random.default_rng(SEED)
results = {}

for cohort in ["Fujiwara", "Discovery"]:
    mat = root["mats"][cohort]
    st = root["stages"][cohort]
    st = (st.loc[mat.columns] if hasattr(st, "loc")
          else pd.Series(st.values, index=mat.columns)).astype(int)
    z = mat.sub(mat.mean(axis=1), axis=0).div(mat.std(axis=1).replace(0, np.nan), axis=0)
    idx = np.array(z.index)

    def score(genes):
        g = [x for x in genes if x in z.index]
        return z.loc[g].mean(axis=0)

    out = {}
    for name, gs, size in [("suppressors", suppressors, 178), ("drivers", drivers, 185)]:
        obs = score(gs)
        rho_obs = spearmanr(obs, st.loc[obs.index]).statistic
        null = np.array([spearmanr(z.iloc[rng.permutation(len(idx))[:size]].mean(axis=0)
                                   if False else score(rng.choice(idx, size=size, replace=False)),
                                   st.loc[obs.index]).statistic
                         for _ in range(N_NULL)])
        p_two = (1 + (np.abs(null) >= abs(rho_obs)).sum()) / (N_NULL + 1)
        out[name] = {"rho_obs": float(rho_obs),
                     "null_mean": float(null.mean()), "null_sd": float(null.std()),
                     "null_p2.5": float(np.percentile(null, 2.5)),
                     "null_p97.5": float(np.percentile(null, 97.5)),
                     "obs_percentile": float((null < rho_obs).mean()),
                     "empirical_p_two_sided": float(p_two),
                     "n_null": N_NULL}
        verdict = ("SPECIFIC (observed in null tail; null centred near 0)"
                   if abs(null.mean()) < 0.05 and p_two < 0.05 else
                   "GLOBAL DRIFT ARTIFACT (random sets behave like the observed)" if
                   abs(rho_obs - null.mean()) < 2 * null.std() else "INDETERMINATE")
        out[name]["verdict_N1"] = verdict
    results[cohort] = out
    print(f"\n[{cohort}] random-set null (size 178/185, {N_NULL} sets, seed {SEED})")
    print(pd.DataFrame(out).T.round(4).to_string())

# ---------- N2: covariate adjustment ----------
print("\n[N2] technical covariates")
cohort = "Fujiwara"
mat = root["mats"][cohort]
st = root["stages"][cohort]
st = (st.loc[mat.columns] if hasattr(st, "loc")
      else pd.Series(st.values, index=mat.columns)).astype(int)
z = mat.sub(mat.mean(axis=1), axis=0).div(mat.std(axis=1).replace(0, np.nan), axis=0)
lin = np.power(2.0, mat.astype(float))

cov = pd.DataFrame({
    "genes_detected_deg": int(mat.shape[0]),  # degenerate: disclosed
    "global_meanz": z.mean(axis=0),
    "library_proxy_linsum": lin.sum(axis=0),
    "mean_log2_expr": mat.mean(axis=0),
}, index=mat.columns)


def partial_spearman(x, y, covars):
    d = pd.concat([pd.Series(x, name="x"), pd.Series(y, name="y")] +
                  [pd.Series(c, name=f"c{i}") for i, c in enumerate(covars)], axis=1).dropna()
    rx, ry = rankdata(d.x), rankdata(d.y)
    X = np.column_stack([rankdata(d[f"c{i}"]) for i in range(len(covars))] + [np.ones(len(d))])
    resid = rx - X @ np.linalg.lstsq(X, rx, rcond=None)[0]
    r, p = spearmanr(resid, ry)
    return r, p, len(d)


adj_rows = []
for name, gs in [("suppressors", suppressors), ("drivers", drivers)]:
    s = z.loc[[g for g in gs if g in z.index]].mean(axis=0)
    r0, p0 = spearmanr(s, st.loc[s.index])
    adj_rows.append({"cohort": "Fujiwara (n=213)", "score": name, "adjustment": "none",
                     "rho": r0, "p": p0, "n": len(s)})
    for cname in ["global_meanz", "library_proxy_linsum", "mean_log2_expr"]:
        c = cov[cname].loc[s.index]
        rc, pc = spearmanr(s, c)
        r, p, n = partial_spearman(s.to_numpy(), st.loc[s.index].to_numpy(), [c.to_numpy()])
        adj_rows.append({"cohort": "Fujiwara (n=213)", "score": name,
                         "adjustment": f"partial ctrl {cname} (corr score~cov rho={rc:.3f})",
                         "rho": r, "p": p, "n": n})
    allc = [cov[c].loc[s.index].to_numpy() for c in
            ["global_meanz", "library_proxy_linsum", "mean_log2_expr"]]
    r, p, n = partial_spearman(s.to_numpy(), st.loc[s.index].to_numpy(), allc)
    adj_rows.append({"cohort": "Fujiwara (n=213)", "score": name,
                     "adjustment": "partial ctrl ALL THREE proxies", "rho": r, "p": p, "n": n})

# real raw-count covariates: 58-patient baseline subset
raw = pd.read_csv("data/kamzolas/Fujiwara_dataset/raw_counts_fujiwara.csv", index_col=0)
emap = pd.read_csv("data/kamzolas/ensembl_mapping.tsv", sep="\t").dropna(subset=["external_gene_name"])
smap = dict(zip(emap.ensembl_gene_id.str.split(".").str[0],
                emap.external_gene_name.str.upper().str.strip()))
raw.index = raw.index.astype(str).str.split(".").str[0].map(smap)
raw = raw[raw.index.notna()].groupby(level=0).mean()
base_cols = [c for c in raw.columns if c.endswith("_1")]
def to_locked(c):  # HUnafld001_1 -> GSE192959.TUnafld001 (notebook cell-68 convention)
    pid = c[:-2]
    return f"GSE192959.{pid.replace('HUnafld', 'TUnafld')}"
locked_of_raw = {c: to_locked(c) for c in base_cols if to_locked(c) in mat.columns}
lib = np.log2(raw[base_cols].sum(axis=0) + 1)
gd = (raw[base_cols] > 0).sum(axis=0)
for name, gs in [("suppressors", suppressors), ("drivers", drivers)]:
    sub_cols = list(locked_of_raw.values())
    raw_cols = list(locked_of_raw.keys())
    s = z.loc[[g for g in gs if g in z.index]].mean(axis=0)[sub_cols]
    stag = st.loc[sub_cols]
    r0, p0 = spearmanr(s, stag)
    adj_rows.append({"cohort": f"Fujiwara raw-count baseline subset (n={len(sub_cols)})",
                     "score": name, "adjustment": "none", "rho": r0, "p": p0, "n": len(s)})
    rl, pl = spearmanr(s, lib.loc[raw_cols])
    rg, pg = spearmanr(s, gd.loc[raw_cols])
    for cn, cv, rc in [("log library size (raw counts)", lib, rl),
                       ("genes detected (raw counts)", gd, rg)]:
        cc = cv.loc[raw_cols].to_numpy()
        r, p, n = partial_spearman(s.to_numpy(), stag.to_numpy(), [cc])
        adj_rows.append({"cohort": f"Fujiwara raw-count baseline subset (n={n})",
                         "score": name,
                         "adjustment": f"partial ctrl {cn} (corr score~cov rho={rc:.3f})",
                         "rho": r, "p": p, "n": n})
adj = pd.DataFrame(adj_rows)
adj.to_csv(f"{OUT}/fuji_covariate_adjustment.csv", index=False)
print(adj.round(4).to_string(index=False))


def md5(p):
    import hashlib
    h = hashlib.md5()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


json.dump(results, open(f"{OUT}/random_set_null.json", "w"), indent=2)
json.dump({"adjustments": adj.to_dict("records"),
           "note": "genes_detected degenerate in locked build (12,537/12,537, disclosed); "
                   "proxies on n=213 + raw-count covariates on n=58 baseline subset; "
                   "WS17 precedent in this cohort: genes-detected~FPS rho=0.317, "
                   "timepoint effect 0.005->0.058 after adjustment"},
          open(f"{OUT}/covariate_adjustment_summary.json", "w"), indent=2, default=float)
for out_f, ins in [("random_set_null.json", ["results/ws15/locked_data.pkl",
                                             "data/ferroptosis_driver_ferrdb.csv",
                                             "data/ferroptosis_suppressor_ferrdb.csv"]),
                   ("fuji_covariate_adjustment.csv", ["results/ws15/locked_data.pkl",
                                                      "data/kamzolas/Fujiwara_dataset/raw_counts_fujiwara.csv",
                                                      "data/kamzolas/ensembl_mapping.tsv"])]:
    json.dump({"output": out_f,
               "derived_from": [{"path": p, "md5": md5(p), "bytes": os.path.getsize(p)} for p in ins],
               "script": "scripts/python/51_ws30c_fuji_null_covariates.py",
               "git_commit": os.popen("git rev-parse --short HEAD").read().strip(),
               "timestamp": datetime.now(timezone.utc).isoformat(),
               "random_seed": SEED, "n_null": N_NULL},
              open(f"{OUT}/{out_f}.provenance.json", "w"), indent=2)
print("\nDONE")
