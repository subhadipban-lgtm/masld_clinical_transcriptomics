"""Phase 1 decisive test — pre-specified protocol (see conversation record).

Fixed decisions recorded before running:
- Discovery = the 349 in data/discovery_cohort_349.csv, limma ~ fibrosis_group,
  NO age covariate (age unavailable for 216/349).
- T1 ranked list comes from that fit only.
Seed 42, 10,000 permutations. FerrDb full sets, dual-annotated removed.
"""
import json, zipfile, sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats as st

ROOT = Path("/Users/subhadipbanerjee/masld-revision")
KAM = ROOT / "data/kamzolas"
OUT = ROOT / "results/decisive_test"
SEED = 42
NPERM = 10000
rng = np.random.default_rng(SEED)

def quantile_normalize(df):
    rank_mean = df.stack().groupby(df.rank(method="first").stack().astype(int)).mean()
    out = df.rank(method="min").stack().astype(int).map(rank_mean).unstack()
    return out

# ---------- gene sets: full FerrDb V2, dual-annotated removed ----------
drivers_all = pd.read_csv(ROOT/"data/ferroptosis_driver_ferrdb.csv")["symbol"].astype(str).str.strip().tolist()
suppressors_all = pd.read_csv(ROOT/"data/ferroptosis_suppressor_ferrdb.csv")["symbol"].astype(str).str.strip().tolist()
dual = sorted(set(drivers_all) & set(suppressors_all))
drivers = [g for g in drivers_all if g not in dual]
suppressors = [g for g in suppressors_all if g not in dual]
print(f"FerrDb: {len(drivers_all)} drivers, {len(suppressors_all)} suppressors, {len(dual)} dual-annotated removed")

# ---------- discovery cohort (349) raw counts -> log2 CPM ----------
disc_meta = pd.read_csv(ROOT/"data/discovery_cohort_349.csv")
disc_meta = disc_meta.set_index("sample_id")
ent = pd.read_csv(KAM/"gene_symbols_to_entrez.tsv", sep="\t")
ent = ent.dropna(subset=["entrez_id","gene_symbol"]).drop_duplicates("entrez_id")
ent_map = dict(zip(ent["entrez_id"].astype(int).astype(str), ent["gene_symbol"].astype(str)))
gses = ["GSE135251","GSE130970","GSE185051"]
mats = []
for g in gses:
    f = next((ROOT/"zenodo_upload/raw_data").glob(g+"*"))
    df = pd.read_csv(f, sep="\t", index_col=0)
    df.index = df.index.astype(str)
    mats.append(df)
common = mats[0].index
for m in mats[1:]:
    common = common.intersection(m.index)
M = pd.concat([m.loc[common] for m in mats], axis=1)
M.index = [ent_map.get(i, np.nan) for i in M.index]
M = M[M.index.notna()]
M = M.loc[~M.index.duplicated(keep="first")]
M = M[disc_meta.index]
lib = M.sum(axis=0) / 1e6
cpm = M / lib
keep = (cpm > 1).mean(axis=1) >= 0.10
Mlog = np.log2(cpm[keep] + 1)
print(f"Discovery after CPM>1 in >=10%: {Mlog.shape}")

# ---------- external cohorts ----------
ens = pd.read_csv(KAM/"ensembl_mapping.tsv", sep="\t")
sym_dict = dict(zip(ens["ensembl_gene_id"].astype(str).str.strip(), ens["external_gene_name"].astype(str).str.strip()))

# Fujiwara n=213 (log2 GEO expression; counts unavailable for the 213-sample series — deviation noted)
fuj = pd.read_csv(ROOT/"data/expression_matrix.csv", index_col=0)
fuj_meta = pd.read_csv(ROOT/"data/metadata_with_ferroptosis_scores.csv")
fuj_meta["col"] = fuj_meta["dataset"] + "." + fuj_meta["title"]
fuj_meta = fuj_meta.set_index("col")
fuj_meta["stage"] = pd.to_numeric(fuj_meta["fibrosis stage:ch1"], errors="coerce")
keepc = [c for c in fuj.columns if c in fuj_meta.index]
fuj = fuj[keepc]; fm = fuj_meta.loc[keepc]
det = (fuj > np.log2(2)).mean(axis=1) >= 0.10   # detection filter analogue of CPM>1
fuj = fuj[det]
fuj_n = int(fm["stage"].notna().sum())

with zipfile.ZipFile(KAM/"EPoS_dataset/EPoS_counts.tsv.zip") as z:
    with z.open("EPoS_counts.tsv") as f:
        epos = pd.read_csv(f, sep="\t", index_col=0)
epos.columns = epos.columns.str.strip()
epos_meta = pd.read_csv(KAM/"EPoS_dataset/epos_metadata.csv").set_index("GEO_ID")
epos = epos[[c for c in epos.columns if c in epos_meta.index]]
epos.index = [sym_dict.get(str(i).strip(), str(i).strip()) for i in epos.index]
epos = epos.loc[~epos.index.duplicated(keep="first")]
cpmE = epos / (epos.sum(axis=0)/1e6)
eposL = np.log2(cpmE[(cpmE>1).mean(axis=1)>=0.10] + 1)

ucam = pd.read_csv(KAM/"ucam_sanyal/counts_matrix.csv", index_col=0)
ucam_meta = pd.read_csv(KAM/"ucam_sanyal/metadata.csv").set_index("Sample name")
ucam = ucam[[c for c in ucam.columns if c in ucam_meta.index]]
ucam.index = [sym_dict.get(str(i).strip(), str(i).strip()) for i in ucam.index]
ucam = ucam.loc[~ucam.index.duplicated(keep="first")]
cpmU = ucam / (ucam.sum(axis=0)/1e6)
ucamL = np.log2(cpmU[(cpmU>1).mean(axis=1)>=0.10] + 1)

universe = sorted(set(Mlog.index) & set(fuj.index) & set(eposL.index) & set(ucamL.index))
print("Shared gene universe:", len(universe))

D = quantile_normalize(Mlog.loc[universe])
F = quantile_normalize(fuj.loc[universe])
E = quantile_normalize(eposL.loc[universe])
U = quantile_normalize(ucamL.loc[universe])

dr_u = [g for g in drivers if g in universe]
su_u = [g for g in suppressors if g in universe]
print(f"In universe: {len(dr_u)} drivers, {len(su_u)} suppressors")

D.to_pickle(OUT/"discovery_qnorm.pkl")
for name, df in [("fujiwara",F),("epos",E),("ucam",U)]:
    df.to_pickle(OUT/f"{name}_qnorm.pkl")

def zscores(df):
    mu = df.mean(axis=1); sd = df.std(axis=1, ddof=1)
    z = df.sub(mu, axis=0).div(sd.replace(0, np.nan), axis=0)
    return z.fillna(0.0)  # constant genes carry no information; contribute 0

def set_scores(zdf, dr, su):
    d = zdf.loc[dr].mean(axis=0); s = zdf.loc[su].mean(axis=0)
    return d, s, d - s

def spearman_stage(scores, stages):
    m = stages.notna()
    r = st.spearmanr(scores[m], stages[m])
    return float(r.statistic), float(r.pvalue), int(m.sum())

def coupling_null(zdf, dr, su, nperm, rng):
    Z = zdf.to_numpy()
    gidx = {g:i for i,g in enumerate(zdf.index)}
    d_idx = np.array([gidx[g] for g in dr]); s_idx = np.array([gidx[g] for g in su])
    dsc = Z[d_idx].mean(axis=0); ssc = Z[s_idx].mean(axis=0)
    obs = float(np.corrcoef(dsc, ssc)[0,1])
    # mean-expression-matched null: bin all genes into deciles, draw sets matching
    # the per-decile composition of the real sets, disjoint
    meanexp = Z.mean(axis=1)
    order = np.argsort(meanexp)
    bins = np.empty(len(meanexp), dtype=int)
    bins[order] = np.minimum((np.arange(len(meanexp)) / len(meanexp) * 10).astype(int), 9)
    bin_members = [np.where(bins == b)[0] for b in range(10)]
    def draw_matched(idx_set, exclude):
        counts = np.bincount(bins[idx_set], minlength=10)
        chosen = []
        excl = set(exclude.tolist())
        for b in np.nonzero(counts)[0]:
            pool = [i for i in bin_members[b] if i not in excl]
            take = min(counts[b], len(pool))
            chosen.append(rng.choice(pool, size=take, replace=False))
        return np.concatenate(chosen)
    null = np.empty(nperm)
    for i in range(nperm):
        rd = draw_matched(d_idx, s_idx)
        rs = draw_matched(s_idx, rd)
        null[i] = np.corrcoef(Z[rd].mean(axis=0), Z[rs].mean(axis=0))[0,1]
    emp_p = float((np.sum(np.abs(null) >= abs(obs)) + 1) / (nperm + 1))
    return obs, float(null.mean()), emp_p, null

results = {"protocol": {
    "seed": SEED, "n_permutations": NPERM,
    "discovery_n": int(D.shape[1]), "universe_size": len(universe),
    "ferrdb_drivers_raw": len(drivers_all), "ferrdb_suppressors_raw": len(suppressors_all),
    "dual_annotated_removed": dual,
    "drivers_in_universe": len(dr_u), "suppressors_in_universe": len(su_u),
    "fujiwara_n_staged": fuj_n, "epos_n": int(E.shape[1]), "ucam_n": int(U.shape[1]),
    "fixed_decisions": [
        "Discovery = 349 in data/discovery_cohort_349.csv; limma ~ fibrosis_group with NO age covariate (age unavailable for 216/349)",
        "T1 ranked list from that fit only",
        "Fujiwara 213: GEO log2 expression used (no raw counts exist for the 213-sample series); detection filter >log2(2) in >=10% as CPM>1 analogue"],
}}

# T3 + T4 discovery
zD = zscores(D)
dsc, ssc, ax = set_scores(zD, dr_u, su_u)
stages = disc_meta["fibrosis_stage"].reindex(D.columns)
T3 = {}
for nm, sc in [("driver", dsc), ("suppressor", ssc), ("axis", ax)]:
    rho, p, n = spearman_stage(pd.Series(sc, index=D.columns), stages)
    T3[nm] = {"rho": rho, "p": p, "n": n}
    per_stage = pd.Series(sc, index=D.columns).groupby(stages).median().to_dict()
    T3[nm]["per_stage_median"] = {str(k): float(v) for k, v in per_stage.items()}
obs, null_mean, emp_p, nullD = coupling_null(zD, dr_u, su_u, NPERM, rng)
T4 = {"observed_r": obs, "null_mean": null_mean, "delta_r": obs - null_mean, "empirical_p": emp_p}
results["T3_discovery"] = T3; results["T4_discovery"] = T4
print("T3:", T3); print("T4:", T4)

# T5 external
ext = {}
ext_defs = [("Fujiwara", F, fm["stage"].reindex(F.columns)),
            ("EPoS", E, pd.to_numeric(epos_meta["Fibrosis.stage"], errors="coerce").reindex(E.columns)),
            ("UCAM_Sanyal", U, pd.to_numeric(ucam_meta["Fibrosis"], errors="coerce").reindex(U.columns))]
for nm, df, stage in ext_defs:
    z = zscores(df)
    d, s, a = set_scores(z, dr_u, su_u)
    t3 = {}
    for snm, sc in [("driver", d), ("suppressor", s), ("axis", a)]:
        rho, p, n = spearman_stage(pd.Series(np.asarray(sc), index=df.columns), stage)
        t3[snm] = {"rho": rho, "p": p, "n": n}
    o, nm_mean, ep, _ = coupling_null(z, dr_u, su_u, NPERM, rng)
    ext[nm] = {"n": int(df.shape[1]), "n_stage": t3["driver"]["n"], "T3": t3,
               "T4": {"observed_r": o, "null_mean": nm_mean, "delta_r": o-nm_mean, "empirical_p": ep}}
    print(nm, ext[nm])
results["T5_external"] = ext

with open(OUT/"phase1_python_results.json","w") as f:
    json.dump(results, f, indent=1, default=float)

# limma T1 fit in R next; write inputs
D.T.to_pickle(OUT/"discovery_for_limma.pkl")
json.dump({"drivers": dr_u, "suppressors": su_u}, open(OUT/"gene_sets.json","w"))
print("saved phase1_python_results.json")
