#!/usr/bin/env python3
"""
WS28 analysis 2.4 — ferroptosis co-expression network WITH its permutation null.

Pre-committed rules (printed before any result):
  R1: nodes = FerrDb filtered drivers (264) + suppressors (238) intersected with the locked
      12,537-gene discovery matrix. Edge = |Spearman rho| > 0.5 across the 349 samples.
  R2: NULL = 1000 permutations, each gene's sample vector independently permuted (seed 42),
      same edge rule. The network counts as EXCESS co-expression only if the observed edge
      count exceeds the 95th percentile of the null. Otherwise the edge count is attributable
      to shared marginal behaviour alone and no module claim may be made.
  R3: modules = greedy modularity on the observed graph (reported only if R2 passes).
      Driver-suppressor mixing = observed cross-type edge fraction vs the random-wiring
      expectation 2*nD*nS/(n*(n-1)), tested hypergeometrically.
Output -> results/ws28_network/
"""
import json
import os
import pickle
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from statsmodels.stats.multitest import multipletests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 42
N_PERM = 1000
RHO = 0.5
OUT = "results/ws28_network"
os.makedirs(OUT, exist_ok=True)

print("R1: nodes = FerrDb 264+238 intersected with locked matrix; edge = |Spearman|>0.5 (n=349).")
print("R2: null = 1000 independent per-gene permutations (seed 42); excess only if obs > null P95.")
print("R3: modules only if R2 passes; mixing vs random-wiring expectation, hypergeometric test.")


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
            "script": "scripts/python/44_ws28_network.py",
            "git_commit": os.popen("git rev-parse --short HEAD").read().strip(),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "random_seed": SEED, "n_permutations": N_PERM}
    if extra:
        prov.update(extra)
    with open(f"{OUT}/{output}.provenance.json", "w") as f:
        json.dump(prov, f, indent=2)


rng = np.random.default_rng(SEED)

with open("results/ws15/locked_data.pkl", "rb") as f:
    root = pickle.load(f)
mat = root["mats"]["Discovery"]
drivers = set(pd.read_csv("data/ferroptosis_driver_ferrdb.csv")["symbol"].str.upper().str.strip())
suppressors = set(pd.read_csv("data/ferroptosis_suppressor_ferrdb.csv")["symbol"].str.upper().str.strip())
assert len(drivers) == 264 and len(suppressors) == 238

genes = sorted((drivers | suppressors) & set(mat.index))
types = {g: ("driver" if g in drivers and g not in suppressors else
             "suppressor" if g in suppressors and g not in drivers else "both") for g in genes}
nD = sum(1 for g in genes if types[g] == "driver")
nS = sum(1 for g in genes if types[g] == "suppressor")
nB = sum(1 for g in genes if types[g] == "both")
X = mat.loc[genes].to_numpy(float)
n, p = X.shape
print(f"\nnodes: {n} ({nD} driver, {nS} suppressor, {nB} in both sets) | samples: {p}")


def edge_count_and_rho(A):
    R = np.corrcoef(A)
    iu = np.triu_indices(n, 1)
    r = R[iu]
    return (np.abs(r) > RHO).sum(), r, iu


# observed
Rk = np.apply_along_axis(rankdata, 1, X)
obs_edges, rhos, iu = edge_count_and_rho(Rk)
# p-values for observed rhos (t approximation), BH across all pairs
t = np.abs(rhos) * np.sqrt((p - 2) / np.maximum(1 - rhos ** 2, 1e-12))
from scipy.stats import t as tdist
pv = 2 * tdist.sf(t, p - 2)
padj_min = multipletests(pv, method="fdr_bh")[1][np.abs(rhos) > RHO].max() if obs_edges else np.nan
print(f"observed edges |rho|>0.5: {obs_edges} | max BH-p among edges: {padj_min:.2e}")

# null (R2)
null_counts = np.zeros(N_PERM)
for b in range(N_PERM):
    Pm = np.empty_like(Rk)
    for i in range(n):
        Pm[i] = Rk[i][rng.permutation(p)]
    null_counts[b] = (np.abs(np.corrcoef(Pm))[iu] > RHO).sum()
p95 = np.percentile(null_counts, 95)
excess = obs_edges > p95
z = (obs_edges - null_counts.mean()) / null_counts.std() if null_counts.std() > 0 else float("inf")
print(f"null: mean={null_counts.mean():.2f} P95={p95:.1f} max={null_counts.max():.0f} | "
      f"observed z={z:.1f} -> R2 verdict: {'EXCESS - module analysis permitted' if excess else 'NOT excess - no module claim'}")

np.savetxt(f"{OUT}/null_edge_counts.csv", null_counts, fmt="%d",
           header="null_edge_count", comments="")

stats_out = {
    "rules": "R1 |rho|>0.5; R2 per-gene permutation null, excess iff obs>P95; R3 modules conditional",
    "nodes": n, "n_driver": nD, "n_suppressor": nS, "n_both": nB, "samples": p,
    "observed_edges": int(obs_edges),
    "null_mean": float(null_counts.mean()), "null_P95": float(p95),
    "null_max": float(null_counts.max()), "z": float(z), "R2_excess": bool(excess),
    "max_BH_p_among_edges": float(padj_min) if obs_edges else None,
    "seed": SEED, "permutations": N_PERM,
    "analysis_date": datetime.now(timezone.utc).isoformat(),
}

# edges table
ii, jj = iu
mask = np.abs(rhos) > RHO
edges = pd.DataFrame({"gene1": [genes[a] for a in ii[mask]],
                      "gene2": [genes[b] for b in jj[mask]],
                      "rho": rhos[mask],
                      "type1": [types[genes[a]] for a in ii[mask]],
                      "type2": [types[genes[b]] for b in jj[mask]]})
edges["cross_type"] = edges.type1 != edges.type2
edges.to_csv(f"{OUT}/network_edges.csv", index=False)
write_provenance("network_edges.csv", ["results/ws15/locked_data.pkl",
                                       "data/ferroptosis_driver_ferrdb.csv",
                                       "data/ferroptosis_suppressor_ferrdb.csv"])
write_provenance("null_edge_counts.csv", ["results/ws15/locked_data.pkl"])
print(f"edges table: {len(edges)} rows | cross-type edges: {int(edges.cross_type.sum())}")

# R3: mixing vs random wiring (report regardless; interpret only if R2 passed)
n_nodes = n
exp_frac = 2 * nD * nS / (n_nodes * (n_nodes - 1))
obs_frac = edges.cross_type.mean() if len(edges) else np.nan
from scipy.stats import hypergeom
K = int(edges.cross_type.sum()) if len(edges) else 0
# hypergeometric: N_pairs = C(n,2) possible pairs, K_cross = nD*nS driver-suppressor pairs
N_pairs = n_nodes * (n_nodes - 1) // 2
K_cross = nD * nS
mix_p = hypergeom.sf(K - 1, N_pairs, K_cross, len(edges)) if len(edges) else np.nan
print(f"mixing: observed cross fraction {obs_frac:.3f} vs random-wiring expectation {exp_frac:.3f} "
      f"| hypergeometric p={mix_p:.3e}")
stats_out["mixing"] = {"observed_fraction": float(obs_frac) if len(edges) else None,
                       "expected_fraction": float(exp_frac),
                       "cross_edges": int(K), "total_edges": int(len(edges)),
                       "hypergeom_p": float(mix_p) if len(edges) else None}

# modules (only meaningful if R2 excess)
if excess and obs_edges > 0:
    import networkx as nx
    G = nx.Graph()
    for r in edges.itertuples():
        G.add_edge(r.gene1, r.gene2, weight=abs(r.rho))
    from networkx.algorithms.community import greedy_modularity_communities
    np.random.seed(SEED)
    comms = greedy_modularity_communities(G, weight="weight")
    Q = nx.algorithms.community.modularity(G, comms, weight="weight")
    comp = [len(c) for c in sorted(comms, key=len, reverse=True)]
    print(f"modules: {len(comms)} communities, modularity Q={Q:.3f}, sizes (top 10): {comp[:10]}")
    stats_out["modules"] = {"n_communities": len(comms), "modularity_Q": float(Q),
                            "sizes_top10": comp[:10]}
    mod = {}
    for k, c in enumerate(sorted(comms, key=len, reverse=True)):
        for g in c:
            mod[g] = k
    edges["module_gene1"] = edges.gene1.map(mod)
    edges.to_csv(f"{OUT}/network_edges.csv", index=False)
    pd.DataFrame({"gene": list(mod), "module": list(mod.values()),
                  "type": [types[g] for g in mod]}).to_csv(f"{OUT}/network_modules.csv", index=False)
    write_provenance("network_modules.csv", [f"{OUT}/network_edges.csv"])
    # figure
    fig, ax = plt.subplots(figsize=(9, 9))
    pos = nx.spring_layout(G, seed=SEED, k=1.2 / np.sqrt(len(G)))
    cmap = plt.get_cmap("tab20")
    nx.draw_networkx_edges(G, pos, alpha=0.2, ax=ax)
    nx.draw_networkx_nodes(G, pos, node_size=28,
                           node_color=[cmap(mod[g] % 20) for g in G], ax=ax)
    ax.set_title(f"Ferroptosis co-expression network (|rho|>0.5, n={p}; "
                 f"{int(obs_edges)} edges, Q={Q:.2f})")
    ax.axis("off")
    plt.tight_layout()
    fig.savefig(f"{OUT}/fig_network.png", dpi=300)
    plt.close(fig)
    write_provenance("fig_network.png", [f"{OUT}/network_edges.csv"])

with open(f"{OUT}/stats_ws28_network.json", "w") as f:
    json.dump(stats_out, f, indent=2, default=float)
write_provenance("stats_ws28_network.json", ["results/ws15/locked_data.pkl",
                                             "data/ferroptosis_driver_ferrdb.csv",
                                             "data/ferroptosis_suppressor_ferrdb.csv"])
print("\nDONE — results/ws28_network/")
