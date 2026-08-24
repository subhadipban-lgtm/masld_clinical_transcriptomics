#!/usr/bin/env python3
"""
WS27 — ferroptosis follow-ups in the feasibility-doc priority order.

2.6  Per-adjacent-stage-pair GSEA (DGE pre-computed by scripts/R/42_ws27_stagepair_dge.R,
     limma-trend on the locked WS15 log2CPM discovery matrix).
2.8  Paired-biopsy ferroptosis score change by trajectory (58 patients, raw second-biopsy
     counts -> CPM/log2 -> Ensembl-mapped -> z-scored across all 116 paired samples).
2.9  Driver-minus-suppressor balance score vs stage (difference, not ratio).

Pre-committed rules (stated before any result is computed):
  R1 (2.6): a ferroptosis programme "emerges" at the first adjacent pair, in the order
      F0vF1, F1vF2, F2vF3, F3vF4, with BH-adjusted GSEA p < 0.05 (family = 4 pairs x 2 sets)
      and NES > 0. F3vF4 (n = 73/16) is low-powered: a null there is NOT evidence of absence.
  R2 (2.8): trajectory groups from delta_fibrosis (>0 progressor, <0 regressor, =0 stable).
      Primary tests: (a) Kruskal-Wallis of delta score across groups; (b) Spearman of delta
      score vs delta_fibrosis (continuous). All results reported whatever they show.
      Baseline-stage composition reported per group (progression is dominated by baseline
      stage, AGENTS.md section 6).
  R3 (2.9): balance = driver mean-z MINUS suppressor mean-z. The ratio is not used because
      the denominator (a mean-z) crosses zero. Tests: Spearman vs stage; KW across stages.

Run: python3 scripts/python/43_ws27_ferroptosis_followups.py   (after the R DGE step)
"""

import json
import os
import pickle
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import spearmanr, kruskal, wilcoxon, rankdata, norm
from statsmodels.stats.multitest import multipletests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

SEED = 42
PERMS = 1000
OUT = "results/ws27_ferroptosis_followups"
os.makedirs(OUT, exist_ok=True)
MANIFEST = []
rng = np.random.default_rng(SEED)

FILES = {
    "dgedir": OUT,  # stagepair_dge_F{x}vF{y}.csv from the R step
    "drivers": "data/ferroptosis_driver_ferrdb.csv",
    "suppressors": "data/ferroptosis_suppressor_ferrdb.csv",
    "raw_paired": "data/kamzolas/Fujiwara_dataset/raw_counts_fujiwara.csv",
    "ensmap": "data/kamzolas/ensembl_mapping.tsv",
    "ws2_paired": "results/ws1_signature/ws2_paired_signature_scores.csv",
    "ws26_scores": "results/ws26_ferroptosis_additional/gsva_scores_discovery.csv",
}

RULES = """PRE-COMMITTED RULES (see module docstring):
R1: emergence = first adjacent pair (F0vF1->F3vF4) with BH p<0.05 (8-test family) and NES>0; F3vF4 null is low-power, not absence.
R2: groups from delta_fibrosis (>0 prog, <0 regr, =0 stable); primary KW + continuous Spearman; baseline-stage composition reported.
R3: balance = driver - suppressor mean-z (ratio rejected: denominator crosses zero)."""
print(RULES)


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
        "script": "scripts/python/43_ws27_ferroptosis_followups.py (+ scripts/R/42_ws27_stagepair_dge.R for 2.6 DGE)",
        "git_commit": os.popen("git rev-parse --short HEAD").read().strip(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "random_seed": SEED, "n_permutations": PERMS,
    }
    if extra:
        prov.update(extra)
    with open(f"{OUT}/{output}.provenance.json", "w") as f:
        json.dump(prov, f, indent=2)


def log_load(path, rows, note):
    MANIFEST.append({"path": path, "rows_in_file": rows, "note": note})


def gsea(ranked_genes, weights, gene_set, n_perm=PERMS, rng=rng):
    """Weighted running-sum GSEA on a descending-ranked list. Returns ES, NES, p, hits."""
    hit = np.isin(ranked_genes, list(gene_set))
    nh = hit.sum()
    n = len(hit)
    if nh == 0 or nh == n:
        return 0.0, np.nan, 1.0, int(nh)

    def es_of(h):
        w = np.abs(weights)
        ph = np.cumsum(np.where(h, w, 0.0)); ph /= ph[-1]
        pm = np.cumsum(np.where(~h, 1.0, 0.0)); pm /= pm[-1]
        d = ph - pm
        return d[np.argmax(np.abs(d))]

    E = es_of(hit)
    null = np.array([es_of(rng.permutation(hit)) for _ in range(n_perm)])
    same = null[np.sign(null) == np.sign(E)]
    nes = E / np.abs(same).mean() if len(same) else np.nan
    p = (np.abs(same) >= abs(E)).mean() if len(same) else np.nan
    return float(E), float(nes), float(p), int(nh)


def main():
    stats_out = {"rules": RULES, "seed": SEED, "permutations": PERMS,
                 "analysis_date": datetime.now(timezone.utc).isoformat()}

    drivers = set(pd.read_csv(FILES["drivers"])["symbol"].str.upper().str.strip())
    suppressors = set(pd.read_csv(FILES["suppressors"])["symbol"].str.upper().str.strip())
    assert len(drivers) == 264 and len(suppressors) == 238
    log_load(FILES["drivers"], 264, "FerrDb V2 filtered")
    log_load(FILES["suppressors"], 238, "FerrDb V2 filtered")

    # ================= 2.6: per-stage-pair GSEA =================
    print("\n[2.6] per-adjacent-stage-pair GSEA (rule R1)")
    pairs = [(0, 1), (1, 2), (2, 3), (3, 4)]
    rows26 = []
    dge_inputs = []
    for a, b in pairs:
        fn = f"{OUT}/stagepair_dge_F{a}vF{b}.csv"
        if not os.path.exists(fn):
            raise FileNotFoundError(fn + " — run scripts/R/42_ws27_stagepair_dge.R first")
        d = pd.read_csv(fn)
        dge_inputs.append(fn)
        d = d.dropna(subset=["t"]).sort_values("t", ascending=False)
        d = d[~d.gene.duplicated()]
        for name, gs in [("Drivers", drivers), ("Suppressors", suppressors)]:
            es, nes, p, hits = gsea(d.gene.to_numpy(), d.t.to_numpy(), gs)
            rows26.append({"pair": f"F{a}vF{b}", "set": name, "genes_tested": len(d),
                           "genes_hit": hits, "ES": es, "NES": nes, "p": p})
    g = pd.DataFrame(rows26)
    g["p_adj"] = multipletests(g.p, method="fdr_bh")[1]
    g.to_csv(f"{OUT}/stagepair_gsea_results.csv", index=False)
    write_provenance("stagepair_gsea_results.csv", dge_inputs + [FILES["drivers"], FILES["suppressors"]],
                     extra={"note": "BH across all 8 tests (4 pairs x 2 sets); DGE = limma-trend on locked log2CPM matrix, scripts/R/42_ws27_stagepair_dge.R"})
    print(g.round(4).to_string(index=False))
    for name in ["Suppressors", "Drivers"]:
        sub = g[(g["set"] == name) & (g.p_adj < 0.05) & (g.NES > 0)]
        first = sub.pair.iloc[0] if len(sub) else None
        print(f"  R1 verdict [{name}]: first emerging pair = {first}")
        stats_out[f"2.6_{name}_first_emerging_pair"] = first
    stats_out["2.6_gsea"] = {f"{r['set']}|{r['pair']}":
                             {"NES": r["NES"], "p": r["p"], "p_adj": r["p_adj"], "hits": r["genes_hit"]}
                             for r in g.to_dict("records")}

    # ================= 2.8: paired biopsies by trajectory =================
    print("\n[2.8] paired-biopsy ferroptosis scores by trajectory (rule R2)")
    raw = pd.read_csv(FILES["raw_paired"], index_col=0)
    log_load(FILES["raw_paired"], raw.shape[0], f"{raw.shape[1]} sample columns (58 patients x 2 timepoints)")
    emap = pd.read_csv(FILES["ensmap"], sep="\t")
    emap = emap.dropna(subset=["external_gene_name"])
    smap = dict(zip(emap.ensembl_gene_id.str.split(".").str[0],
                    emap.external_gene_name.str.upper().str.strip()))
    raw.index = raw.index.astype(str).str.split(".").str[0].map(smap)
    raw = raw[raw.index.notna()]
    raw = raw.groupby(level=0).mean()
    ws2 = pd.read_csv(FILES["ws2_paired"], index_col=0)
    log_load(FILES["ws2_paired"], len(ws2), "real clinical deltas + trajectory (verified 2026-08-24)")
    patients = [p for p in ws2.index
                if f"{p}_1" in raw.columns and f"{p}_2" in raw.columns]
    assert len(patients) == 58, f"expected 58 paired patients, got {len(patients)}"
    cols = [f"{p}_1" for p in patients] + [f"{p}_2" for p in patients]
    sub = raw[cols].astype(float)
    cpm = sub.div(sub.sum(axis=0) / 1e6, axis=1)
    logcpm = np.log2(cpm + 1)
    z = logcpm.sub(logcpm.mean(axis=1), axis=0).div(logcpm.std(axis=1).replace(0, np.nan), axis=0)

    def score(gs):
        genes = [x for x in gs if x in z.index]
        return z.loc[genes].mean(axis=0), len(genes)

    drv_1, n_drv = score(drivers)
    sup_1, n_sup = score(suppressors)
    paired = pd.DataFrame({
        "patient": patients,
        "driver_baseline": [drv_1[f"{p}_1"] for p in patients],
        "driver_followup": [drv_1[f"{p}_2"] for p in patients],
        "supp_baseline": [sup_1[f"{p}_1"] for p in patients],
        "supp_followup": [sup_1[f"{p}_2"] for p in patients],
    }).set_index("patient")
    paired["driver_delta"] = paired.driver_followup - paired.driver_baseline
    paired["supp_delta"] = paired.supp_followup - paired.supp_baseline
    paired = paired.join(ws2[["baseline_stage", "delta_fibrosis", "delta_NAS", "progressed"]])
    paired.to_csv(f"{OUT}/paired_ferroptosis_scores.csv")

    # timepoint-order verification: our 649-signature mean-z at _1/_2 must correlate with
    # the REAL ws2 baseline/follow-up scores in the right order
    sig = pd.read_csv("results/ws1_signature/ws1_signature_genes.csv")
    sig_col = sig["GeneSymbol"].astype(str).str.upper().str.strip()
    sig_score, n_sig = score(set(sig_col))
    s1 = np.array([sig_score[f"{p}_1"] for p in patients])
    s2 = np.array([sig_score[f"{p}_2"] for p in patients])
    b = ws2.loc[patients, "baseline_score"].astype(float).to_numpy()
    f_ = ws2.loc[patients, "followup_score"].astype(float).to_numpy()
    r_bb = spearmanr(s1, b).statistic
    r_bf = spearmanr(s1, f_).statistic
    print(f"  timepoint-order check: cor(our _1 signature score, ws2 baseline)={r_bb:.3f} "
          f"vs cor(our _1, ws2 follow-up)={r_bf:.3f} -> _1 is "
          f"{'BASELINE' if r_bb > r_bf else 'FOLLOW-UP??'} (n_sig_genes={n_sig})")
    stats_out["2.8_timepoint_check"] = {"cor_ours1_ws2baseline": r_bb,
                                        "cor_ours1_ws2followup": r_bf,
                                        "_1_is_baseline": bool(r_bb > r_bf)}

    def traj(d):
        return "progressor" if d > 0 else ("regressor" if d < 0 else "stable")

    paired["trajectory"] = paired.delta_fibrosis.map(traj)
    print("  trajectory n:", paired.trajectory.value_counts().to_dict())
    print("  baseline stage by trajectory:",
          {k: v.describe()[["mean", "min", "max"]].round(2).to_dict()
           for k, v in paired.groupby("trajectory").baseline_stage})
    res28 = []
    for col in ["driver_delta", "supp_delta"]:
        groups = [paired[paired.trajectory == t][col] for t in ["progressor", "regressor", "stable"]]
        if all(len(g_) > 0 for g_ in groups) and len(groups) > 2:
            h, p = kruskal(*groups)
        else:
            h, p = np.nan, np.nan
        r, pr = spearmanr(paired[col], paired.delta_fibrosis)
        w, pw = wilcoxon(paired[col.replace("_delta", "_baseline")],
                         paired[col.replace("_delta", "_followup")])
        res28.append({"score": col, "KW_H": h, "KW_p": p,
                      "spearman_vs_delta_fibrosis": r, "spearman_p": pr,
                      "wilcoxon_timepoint_p": pw,
                      "n": len(paired),
                      "n_prog": int((paired.trajectory == "progressor").sum()),
                      "n_regr": int((paired.trajectory == "regressor").sum()),
                      "n_stable": int((paired.trajectory == "stable").sum())})
    r28 = pd.DataFrame(res28)
    r28["KW_p_adj"] = multipletests(r28.KW_p, method="fdr_bh")[1]
    r28.to_csv(f"{OUT}/paired_trajectory_results.csv", index=False)
    write_provenance("paired_ferroptosis_scores.csv",
                     [FILES["raw_paired"], FILES["ensmap"], FILES["drivers"],
                      FILES["suppressors"], FILES["ws2_paired"]],
                     extra={"note": "58 patients x 2 timepoints; counts->CPM->log2(+1); Ensembl mapped via ensembl_mapping.tsv; z across all 116 paired samples; genes used: drivers %d / suppressors %d" % (n_drv, n_sup)})
    write_provenance("paired_trajectory_results.csv", [f"{OUT}/paired_ferroptosis_scores.csv"])
    print(r28.to_string(index=False, float_format=lambda v: f"{v:.3e}" if v < 0.001 else f"{v:.3f}"))
    stats_out["2.8"] = {r["score"]: {"KW_p": r["KW_p"], "spearman_rho": r["spearman_vs_delta_fibrosis"],
                                     "spearman_p": r["spearman_p"], "wilcoxon_p": r["wilcoxon_timepoint_p"]}
                        for r in res28}
    stats_out["2.8_group_n"] = paired.trajectory.value_counts().to_dict()
    stats_out["2.8_genes_used"] = {"drivers": n_drv, "suppressors": n_sup}

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, col, lab in zip(axes, ["driver_delta", "supp_delta"], ["Driver", "Suppressor"]):
        sns.boxplot(data=paired.reset_index(), x="trajectory", y=col,
                    order=["regressor", "stable", "progressor"], ax=ax, hue="trajectory", legend=False)
        ax.axhline(0, ls="--", c="grey", lw=0.8)
        ax.set_title(f"{lab} score change (follow-up - baseline), n=58")
    plt.tight_layout()
    fig.savefig(f"{OUT}/fig_paired_trajectory.png", dpi=300)
    plt.close(fig)
    write_provenance("fig_paired_trajectory.png", [f"{OUT}/paired_ferroptosis_scores.csv"])

    # ================= 2.9: balance score =================
    print("\n[2.9] driver-minus-suppressor balance vs stage (rule R3)")
    s26 = pd.read_csv(FILES["ws26_scores"], index_col=0)
    log_load(FILES["ws26_scores"], len(s26), "WS26 discovery mean-z scores")
    s26["balance"] = s26.driver_score - s26.suppressor_score
    r9, p9 = spearmanr(s26.balance, s26.stage)
    groups = [s26[s26.stage == s].balance for s in range(5)]
    h9, kp9 = kruskal(*groups)
    bal = s26[["balance", "stage"]].copy()
    bal.to_csv(f"{OUT}/balance_score_by_stage.csv")
    pd.DataFrame({"test": ["Spearman_vs_stage", "KruskalWallis"],
                  "stat": [r9, h9], "p": [p9, kp9], "n": [len(s26), len(s26)]}). \
        to_csv(f"{OUT}/balance_score_tests.csv", index=False)
    write_provenance("balance_score_tests.csv", [FILES["ws26_scores"]],
                     extra={"note": "balance = driver mean-z - suppressor mean-z (difference per rule R3)"})
    print(f"  balance vs stage: Spearman rho={r9:.3f} p={p9:.3e} | KW H={h9:.3f} p={kp9:.3e} n={len(s26)}")
    stats_out["2.9"] = {"spearman_rho": r9, "spearman_p": p9, "KW_H": h9, "KW_p": kp9, "n": len(s26)}

    # ================= manifest + stats =================
    pd.DataFrame(MANIFEST).to_csv(f"{OUT}/load_manifest.csv", index=False)
    write_provenance("load_manifest.csv",
                     sorted({m["path"] for m in MANIFEST}))
    with open(f"{OUT}/stats_ws27.json", "w") as f:
        json.dump(stats_out, f, indent=2, default=float)
    write_provenance("stats_ws27.json", [FILES["drivers"], FILES["suppressors"], FILES["raw_paired"],
                                         FILES["ensmap"], FILES["ws2_paired"], FILES["ws26_scores"]] + dge_inputs)
    print("\nDONE — outputs in", OUT)


if __name__ == "__main__":
    main()
