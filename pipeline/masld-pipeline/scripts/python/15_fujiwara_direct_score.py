#!/usr/bin/env python3
"""
WS30 addendum — direct suppressor mean-z score in Fujiwara (no GSEA).

Question (user, 2026-08-25): the T1 Fujiwara GSEA inversion (suppressor NES negative at
every transition) could be a rank/tail artifact. A direct per-sample mean-z score, exactly
as Section 3.13 uses in discovery, is not rank-based.

PRE-COMMITTED READING (stated before computing): if the direct Fujiwara suppressor score
also associates negatively with stage, the inversion is NOT a rank artifact and the
suppressor-stage association fails to replicate in Fujiwara at score level. If it is
positive, the GSEA inversion was a tail artifact and the score-level association replicates.

Also reports measurability: suppressors/drivers present in the locked Fujiwara matrix
index vs actually scored (sd > 0), against the discovery matrix on the identical code path.
Output -> results/ws30/ (fujiwara_direct_score.csv, direct_score_tests.csv,
ferroptosis_measurability.csv, stats json, provenance).
"""
import json
import os
import pickle
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, kruskal

SEED = 42
OUT = "results/ws30"
print("PRE-COMMITTED READING: direct-score negative in Fujiwara -> inversion is real, "
      "not a rank artifact; positive -> GSEA tail artifact.")

with open("results/ws15/locked_data.pkl", "rb") as f:
    root = pickle.load(f)
drivers = set(pd.read_csv("data/ferroptosis_driver_ferrdb.csv")["symbol"].str.upper().str.strip())
suppressors = set(pd.read_csv("data/ferroptosis_suppressor_ferrdb.csv")["symbol"].str.upper().str.strip())

rows, tests, meas = [], [], []
for cohort in ["Discovery", "Fujiwara", "UCAM"]:
    mat = root["mats"][cohort]
    st = root["stages"][cohort]
    st = (st.loc[mat.columns] if hasattr(st, "loc")
          else pd.Series(st.values, index=mat.columns)).astype(int)
    z = mat.sub(mat.mean(axis=1), axis=0).div(
        mat.std(axis=1).replace(0, np.nan), axis=0)
    for name, gs in [("suppressor", suppressors), ("driver", drivers)]:
        in_index = [g for g in gs if g in z.index]
        scored = z.loc[in_index]
        scored = scored[scored.notna().all(axis=1) & (mat.loc[in_index].std(axis=1) > 0)]
        s = scored.mean(axis=0)
        rows.append(pd.DataFrame({"cohort": cohort, "sample_id": s.index,
                                  "score_type": name, "score": s.values,
                                  "stage": st.loc[s.index].values}))
        r, p = spearmanr(s, st.loc[s.index])
        groups = [s[st.loc[s.index] == k] for k in range(5)]
        groups = [g for g in groups if len(g) > 0]
        h, kp = kruskal(*groups) if len(groups) > 2 else (np.nan, np.nan)
        tests.append({"cohort": cohort, "score_type": name, "n": len(s),
                      "genes_scored": scored.shape[0],
                      "spearman_rho_vs_stage": r, "spearman_p": p,
                      "KW_H": h, "KW_p": kp,
                      "mean_F0": s[st.loc[s.index] == 0].mean(),
                      "mean_F4": s[st.loc[s.index] == 4].mean() if (st == 4).any() else np.nan})
        meas.append({"cohort": cohort, "set": name, "filtered_size": len(gs),
                     "in_matrix_index": len(in_index),
                     "scored_sd_gt_0": scored.shape[0]})
per = pd.concat(rows, ignore_index=True)
per.to_csv(f"{OUT}/fujiwara_direct_score.csv", index=False)
t = pd.DataFrame(tests)
t.to_csv(f"{OUT}/direct_score_tests.csv", index=False)
m = pd.DataFrame(meas)
m.to_csv(f"{OUT}/ferroptosis_measurability.csv", index=False)

print("\nmeasurability (filtered -> in locked index -> scored sd>0):")
print(m.to_string(index=False))
print("\ndirect-score tests:")
print(t.round(4).to_string(index=False))
fuji_sup = t[(t.cohort == "Fujiwara") & (t.score_type == "suppressor")].iloc[0]
verdict = ("INVERSION CONFIRMED AT SCORE LEVEL — not a rank artifact"
           if fuji_sup.spearman_rho_vs_stage < 0 else
           "score-level association positive — GSEA inversion was a tail artifact")
print(f"\nREADING: {verdict}")


def md5(p):
    import hashlib
    h = hashlib.md5()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


for out_f, ins in [("fujiwara_direct_score.csv", ["results/ws15/locked_data.pkl",
                                                  "data/ferroptosis_driver_ferrdb.csv",
                                                  "data/ferroptosis_suppressor_ferrdb.csv"]),
                   ("direct_score_tests.csv", [f"{OUT}/fujiwara_direct_score.csv"]),
                   ("ferroptosis_measurability.csv", ["results/ws15/locked_data.pkl",
                                                      "data/ferroptosis_suppressor_ferrdb.csv"])]:
    json.dump({"output": out_f,
               "derived_from": [{"path": p, "md5": md5(p), "bytes": os.path.getsize(p)} for p in ins],
               "script": "scripts/python/50_ws30b_fuji_direct_score.py",
               "git_commit": os.popen("git rev-parse --short HEAD").read().strip(),
               "timestamp": datetime.now(timezone.utc).isoformat(),
               "note": "mean within-sample z of set members with sd>0; identical estimator "
                       "to Section 3.13 / WS26; locked matrices (Fujiwara = GEO log2 "
                       "expression per documented 2.2 deviation)"},
              open(f"{OUT}/{out_f}.provenance.json", "w"), indent=2)

json.dump({"verdict": verdict,
           "fuji_suppressor": {"rho": float(fuji_sup.spearman_rho_vs_stage),
                               "p": float(fuji_sup.spearman_p),
                               "KW_p": float(fuji_sup.KW_p),
                               "genes_scored": int(fuji_sup.genes_scored)},
           "tests": t.to_dict("records"), "measurability": m.to_dict("records")},
          open(f"{OUT}/stats_ws30b_direct_score.json", "w"), indent=2, default=float)
json.dump({"output": "stats_ws30b_direct_score.json",
           "derived_from": [{"path": f"{OUT}/direct_score_tests.csv"}],
           "script": "scripts/python/50_ws30b_fuji_direct_score.py",
           "git_commit": os.popen("git rev-parse --short HEAD").read().strip(),
           "timestamp": datetime.now(timezone.utc).isoformat()},
          open(f"{OUT}/stats_ws30b_direct_score.json.provenance.json", "w"), indent=2)
print("\nDONE")
