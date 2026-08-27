#!/usr/bin/env python3
"""
WS30d — random-set null across ALL seven Table-7 pathways, discovery + UCAM.

PRE-COMMITTED RULE (before computing): a pathway's stage association is SPECIFIC only if
its empirical two-sided p against 1,000 size-matched random gene sets (identical mean-z
estimator, same cohort, seed-derived stream recorded) is < 0.05. Otherwise the row is
NULL-TYPICAL (attributable to genome-wide mean-z drift) and does not survive into Table 7.
The null centre is reported for every pathway as the measured drift. Do not re-specify
after seeing results.
"""
import json
import os
import pickle
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

SEED = 42
N_NULL = 1000
OUT = "results/ws30"
print("PRE-COMMITTED RULE: SPECIFIC iff empirical two-sided p < 0.05 vs 1,000 "
      "size-matched random sets; else NULL-TYPICAL (drift).")

GO = {"apoptosis": "GO:0006915", "autophagy": "GO:0006914", "necroptosis": "GO:0097300",
      "pyroptosis": "GO:0141201", "ferroptosis_GO": "GO:0097707"}
sets = {}
for name, go in GO.items():
    cache = f"data/quickgo/{name}_{go.replace(':', '_')}_human.tsv"
    sets[name] = set(pd.read_csv(cache, sep="\t", usecols=["SYMBOL"]).SYMBOL
                     .dropna().astype(str).str.upper().str.strip())
sets["FerrDb_Drivers"] = set(pd.read_csv("data/ferroptosis_driver_ferrdb.csv")["symbol"]
                             .str.upper().str.strip())
sets["FerrDb_Suppressors"] = set(pd.read_csv("data/ferroptosis_suppressor_ferrdb.csv")["symbol"]
                                 .str.upper().str.strip())

with open("results/ws15/locked_data.pkl", "rb") as f:
    root = pickle.load(f)

rows = []
for cohort in ["Discovery", "UCAM"]:
    mat = root["mats"][cohort]
    st = root["stages"][cohort]
    st = (st.loc[mat.columns] if hasattr(st, "loc")
          else pd.Series(st.values, index=mat.columns)).astype(int)
    z = mat.sub(mat.mean(axis=1), axis=0).div(mat.std(axis=1).replace(0, np.nan), axis=0)
    idx = np.array(z.index)
    rng = np.random.default_rng(SEED)
    for name, gs in sets.items():
        members = [g for g in gs if g in z.index]
        size = len(members)
        obs = z.loc[members].mean(axis=0)
        rho_obs = spearmanr(obs, st.loc[obs.index]).statistic
        null = np.array([spearmanr(z.loc[rng.choice(idx, size=size, replace=False)].mean(axis=0),
                                   st.loc[obs.index]).statistic for _ in range(N_NULL)])
        p_two = (1 + (np.abs(null) >= abs(rho_obs)).sum()) / (N_NULL + 1)
        rows.append({"cohort": cohort, "pathway": name, "n_genes": size,
                     "rho_obs": rho_obs, "null_mean": null.mean(), "null_sd": null.std(),
                     "obs_percentile": (null < rho_obs).mean(),
                     "empirical_p_two_sided": p_two,
                     "verdict": "SPECIFIC" if p_two < 0.05 else "NULL-TYPICAL (drift)"})
r = pd.DataFrame(rows)
r.to_csv(f"{OUT}/random_set_null_pathways.csv", index=False)
print(r.round(4).to_string(index=False))

sig = r[r.verdict == "SPECIFIC"]
print("\nSurviving rows:", sig[["cohort", "pathway", "rho_obs",
                               "empirical_p_two_sided"]].to_dict("records"))


def md5(p):
    import hashlib
    h = hashlib.md5()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


ins = ["results/ws15/locked_data.pkl", "data/ferroptosis_driver_ferrdb.csv",
       "data/ferroptosis_suppressor_ferrdb.csv"] + \
      [f"data/quickgo/{n}_{g.replace(':', '_')}_human.tsv" for n, g in GO.items()]
json.dump({"output": "random_set_null_pathways.csv",
           "derived_from": [{"path": p, "md5": md5(p), "bytes": os.path.getsize(p)} for p in ins],
           "script": "scripts/python/52_ws30d_null_pathways.py",
           "git_commit": os.popen("git rev-parse --short HEAD").read().strip(),
           "timestamp": datetime.now(timezone.utc).isoformat(),
           "random_seed": SEED, "n_null": N_NULL,
           "rule": "SPECIFIC iff empirical two-sided p < 0.05 vs size-matched null",
           "methods_items_for_text_pass": [
               "The random-set null must be described once in Methods as a general guard on "
               "every mean-z pathway score in the paper (1,000 size-matched sets, identical "
               "estimator, empirical two-sided p).",
               "Disclose: genes-detected is degenerate in the locked Fujiwara matrix "
               "(12,537 everywhere, no zero entries - post-normalisation GEO log2), so true "
               "library-size/genes-detected adjustment is impossible there and the n=40 "
               "raw-count baseline subset was used instead."]},
          open(f"{OUT}/random_set_null_pathways.csv.provenance.json", "w"), indent=2)
json.dump({"surviving": sig[["cohort", "pathway", "rho_obs", "empirical_p_two_sided"]]
           .to_dict("records"),
           "all": r.to_dict("records")},
          open(f"{OUT}/stats_ws30d_null_pathways.json", "w"), indent=2, default=float)
print("\nDONE")
