"""WS5: trial benchmark. CAVEAT: embedding acceptance criteria failed (effective rank ~2.9,
mean cosine ~0.91, frac>0.99 ~50%) - results reported under the pre-agreed caveat.
Score = cosine(drug embedding, mean of top-20 signature-gene embeddings)."""
import json, numpy as np, pandas as pd, networkx as nx
from scipy import stats
from sklearn.metrics import roc_auc_score
G=nx.read_gexf('data/kg/masld_kg_v3.gexf')
# freeze node order to the trained graph BEFORE any additions
orig_nodes=sorted(G.nodes()); orig_idx={n:i for i,n in enumerate(orig_nodes)}
sig=pd.read_csv('results/ws1_signature/ws1_signature_genes.csv').sort_values('adj.P.Val')
top20=sig.GeneSymbol.head(20).tolist()
tr=pd.read_csv('data/trial_outcomes.csv')
Z=np.load('results/kg_v3_embeddings.npy')
nodes=sorted(G.nodes()); idx={n:i for i,n in enumerate(nodes)}
dt=pd.read_csv('data/kg/drug_target_merged.csv')
added=[]
for d in tr.drug.unique():
    if d not in orig_idx:
        tg=dt[dt.drug_name.str.lower()==d.lower()].target_gene.tolist()
        tg=[t for t in tg if t in orig_idx]
        added.append((d,len(set(tg)),d in orig_idx))
print('trial drugs NOT in trained graph (targets available if added later):',added)
idx=orig_idx  # embeddings index ONLY the trained node set
mean_sig=Z[[idx[g] for g in top20 if g in idx]].mean(axis=0)
mean_sig/=np.linalg.norm(mean_sig)
rows=[]
for _,r in tr.iterrows():
    d=r.drug
    if d in idx:
        v=Z[idx[d]]; v=v/np.linalg.norm(v)
        sc=float(v@mean_sig)
    else:
        sc=np.nan  # not embedded: excluded from evaluation (no post-hoc scoring)
    rows.append({'drug':d,'outcome':r.phase3_histology_outcome,'trial':r.trial,'score':sc,'in_trained_graph':d in idx})
df=pd.DataFrame(rows); print(df.to_string())
ev=df[df.outcome.isin(['success','failure']) & df.score.notna()]
y=(ev.outcome=='success').astype(int).values; s=ev.score.values
print('n evaluated:',len(ev),'success:',int(y.sum()),'failure:',int((1-y).sum()))
auc=roc_auc_score(y,s); print('Trial AUROC:',round(auc,3))
rng=np.random.default_rng(42); bs=[]
for _ in range(2000):
    i=rng.integers(0,len(s),len(s))
    if len(set(y[i]))<2: continue
    bs.append(roc_auc_score(y[i],s[i]))
ci=[float(np.percentile(bs,2.5)),float(np.percentile(bs,97.5))]
print('bootstrap 95% CI:',[round(c,3) for c in ci])
gate='headline' if auc>=0.70 else ('exploratory' if auc>=0.60 else 'drop_recommendation_language')
out={'trial_AUROC':float(auc),'CI95':ci,'n_evaluated':int(len(ev)),'n_success':int(y.sum()),
 'n_failure':int((1-y).sum()),'n_mixed_excluded':int((df.outcome=='mixed').sum()),
 'n_drugs_not_in_trained_graph':int((~df.in_trained_graph).sum()),
 'caveat':'embedding acceptance criteria FAILED (effective rank ~2.9/32, mean cosine ~0.91, frac>0.99 ~50%); drug features were all-zero (no SMILES/ATC available)',
 'pre_committed_conclusion':gate,'per_drug':rows}
json.dump(out,open('results/trial_benchmark.json','w'),indent=1)
