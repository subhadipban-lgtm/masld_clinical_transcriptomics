#!/usr/bin/env python3
"""
WS28 analysis 2.7 — cell-death pathway stage association vs ferroptosis.

Gene sets: QuickGO REST (descendants, human, 2026-08-24 download) for
  apoptosis GO:0006915, autophagy GO:0006914, programmed necrotic cell death (necroptosis)
  GO:0097300, pyroptotic cell death GO:0141201, ferroptosis GO:0097707.
  MSigDB v2023.2.Hs was NOT used: the download endpoint now redirects to a login page
  (checked 2026-08-24); QuickGO is the stated substitute, IDs recorded above.
  FerrDb drivers/suppressors (264/238) enter via the WS26 scores for method identity.

Pre-committed rule (printed first):
  R1: per pathway, mean-z score across the locked discovery matrix (same estimator as WS26),
      Kruskal-Wallis across F0-F4 and Spearman vs stage; BH across the 7-test family
      (5 GO pathways + FerrDb drivers + FerrDb suppressors). Strongest stage association =
      largest KW -log10 p. All results reported regardless of direction.
Output -> results/ws28_celldeath/ ; gene-set downloads cached in data/quickgo/
"""
import io
import json
import os
import pickle
import urllib.request
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy.stats import kruskal, spearmanr
from statsmodels.stats.multitest import multipletests

OUT = "results/ws28_celldeath"
QD = "data/quickgo"
os.makedirs(OUT, exist_ok=True)
os.makedirs(QD, exist_ok=True)

RULE = ("R1: mean-z per pathway on locked matrix; KW across F0-F4 + Spearman vs stage; "
        "BH across 7 tests; strongest = max KW -log10 p; report all directions.")
print(RULE)

GO_TERMS = {
    "apoptosis": "GO:0006915",
    "autophagy": "GO:0006914",
    "necroptosis": "GO:0097300",
    "pyroptosis": "GO:0141201",
    "ferroptosis_GO": "GO:0097707",
}


def md5(p):
    import hashlib
    h = hashlib.md5()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def write_provenance(output, inputs, extra=None):
    prov = {"output": output,
            "derived_from": [{"path": p, "md5": md5(p), "bytes": os.path.getsize(p)} for p in inputs],
            "script": "scripts/python/45_ws28_celldeath.py",
            "git_commit": os.popen("git rev-parse --short HEAD").read().strip(),
            "timestamp": datetime.now(timezone.utc).isoformat()}
    if extra:
        prov.update(extra)
    with open(f"{OUT}/{output}.provenance.json", "w") as f:
        json.dump(prov, f, indent=2)


# ---- gene sets ----
sets = {}
manifest = []
for name, go in GO_TERMS.items():
    cache = f"{QD}/{name}_{go.replace(':', '_')}_human.tsv"
    if not os.path.exists(cache):
        url = (f"https://www.ebi.ac.uk/QuickGO/services/annotation/downloadSearch"
               f"?goId={go}&goUsage=descendants&taxonId=9606")
        req = urllib.request.Request(url, headers={"Accept": "text/tsv"})
        with urllib.request.urlopen(req, timeout=120) as r:
            data = r.read()
        open(cache, "wb").write(data)
    df = pd.read_csv(cache, sep="\t", usecols=["SYMBOL"])
    sets[name] = set(df.SYMBOL.dropna().astype(str).str.upper().str.strip())
    manifest.append({"path": cache, "rows_in_file": len(df),
                     "note": f"{go} + descendants, human (QuickGO 2026-08-24)"})
    print(f"  {name:15s} {go}: {len(sets[name]):5d} symbols (raw download {len(df)} annotations)")

with open("results/ws15/locked_data.pkl", "rb") as f:
    root = pickle.load(f)
mat = root["mats"]["Discovery"]
stage = root["stages"]["Discovery"]
stage = (stage.loc[mat.columns] if hasattr(stage, "loc")
         else pd.Series(stage.values, index=mat.columns)).astype(int)

z = mat.sub(mat.mean(axis=1), axis=0).div(mat.std(axis=1).replace(0, np.nan), axis=0)


def score(gs):
    g = [x for x in gs if x in z.index]
    return z.loc[g].mean(axis=0), len(g)


rows = []
scored = {}
for name in list(GO_TERMS):
    s, ng = score(sets[name])
    scored[name] = s
    rows.append({"pathway": name, "genes_in_set": len(sets[name]), "genes_used": ng})
# FerrDb arms from the SAME estimator (recomputed here, identical to WS26 method)
drv = set(pd.read_csv("data/ferroptosis_driver_ferrdb.csv")["symbol"].str.upper().str.strip())
sup = set(pd.read_csv("data/ferroptosis_suppressor_ferrdb.csv")["symbol"].str.upper().str.strip())
scored["ferroptosis_drivers_FerrDb"], nd = score(drv)
scored["ferroptosis_suppressors_FerrDb"], ns = score(sup)
rows += [{"pathway": "ferroptosis_drivers_FerrDb", "genes_in_set": 264, "genes_used": nd},
         {"pathway": "ferroptosis_suppressors_FerrDb", "genes_in_set": 238, "genes_used": ns}]

res = []
for name, s in scored.items():
    d = pd.DataFrame({"score": s, "stage": stage}).dropna()
    groups = [d[d.stage == k].score for k in range(5)]
    h, p = kruskal(*groups)
    r, pr = spearmanr(d.score, d.stage)
    res.append({"pathway": name, "n": len(d), "KW_H": h, "KW_p": p,
                "spearman_rho_vs_stage": r, "spearman_p": pr})
R = pd.DataFrame(res).merge(pd.DataFrame(rows), on="pathway")
R["KW_p_adj"] = multipletests(R.KW_p, method="fdr_bh")[1]
R["neglog10_KW_p"] = -np.log10(R.KW_p)
R = R.sort_values("neglog10_KW_p", ascending=False)
R.to_csv(f"{OUT}/celldeath_stage_association.csv", index=False)
write_provenance("celldeath_stage_association.csv",
                 [m["path"] for m in manifest] + ["results/ws15/locked_data.pkl",
                                                  "data/ferroptosis_driver_ferrdb.csv",
                                                  "data/ferroptosis_suppressor_ferrdb.csv"],
                 extra={"rule": RULE, "note": "QuickGO substitute for MSigDB (login-walled); GO IDs recorded in data/quickgo/"})
print(R.to_string(index=False))
best = R.iloc[0]
print(f"\nR1 verdict: strongest stage association = {best.pathway} "
      f"(KW p={best.KW_p:.2e}, adj={best.KW_p_adj:.2e}, rho={best.spearman_rho_vs_stage:.3f})")

pd.DataFrame(manifest).to_csv(f"{OUT}/load_manifest.csv", index=False)
stats = {"rule": RULE,
         "strongest": {"pathway": best.pathway, "KW_p": float(best.KW_p),
                       "KW_p_adj": float(best.KW_p_adj),
                       "spearman_rho": float(best.spearman_rho_vs_stage)},
         "results": {r.pathway: {"KW_p": r.KW_p, "KW_p_adj": r.KW_p_adj,
                                 "rho": r.spearman_rho_vs_stage, "genes_used": r.genes_used}
                     for r in R.itertuples()},
         "go_terms": GO_TERMS, "source": "QuickGO REST, descendants, taxon 9606, downloaded 2026-08-24",
         "analysis_date": datetime.now(timezone.utc).isoformat()}
with open(f"{OUT}/stats_ws28_celldeath.json", "w") as f:
    json.dump(stats, f, indent=2, default=float)
write_provenance("stats_ws28_celldeath.json", [m["path"] for m in manifest])
print("DONE — results/ws28_celldeath/")
