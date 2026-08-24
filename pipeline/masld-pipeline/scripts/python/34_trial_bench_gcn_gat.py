"""Trial benchmark for GCN/GAT embeddings. GATE STATUS: both models FAIL the
effective-rank criterion (>10); benchmark run for transparency only."""
import json, numpy as np, pandas as pd, networkx as nx
from sklearn.metrics import roc_auc_score
G=nx.read_gexf('data/kg/masld_kg_v3.gexf')
sig=pd.read_csv('results/ws1_signature/ws1_signature_genes.csv').sort_values('adj.P.Val')
top20=sig.GeneSymbol.head(20).tolist()
tr=pd.read_csv('data/trial_outcomes.csv')
nodes=sorted(G.nodes()); idx={n:i for i,n in enumerate(nodes)}
for name in ['gcn','gat']:
    Z=np.load(f'results/{name}_embeddings.npy')
    ms=Z[[idx[g] for g in top20 if g in idx]].mean(axis=0); ms/=np.linalg.norm(ms)
    rows=[]
    for _,r in tr.iterrows():
        if r.drug in idx:
            v=Z[idx[r.drug]]; v=v/np.linalg.norm(v); sc=float(v@ms)
        else: sc=float('nan')
        rows.append({'drug':r.drug,'outcome':r.phase3_histology_outcome,'trial':r.trial,'score':sc})
    ev=[r for r in rows if r['outcome'] in ('success','failure') and not np.isnan(r['score'])]
    y=np.array([1 if r['outcome']=='success' else 0 for r in ev]); s=np.array([r['score'] for r in ev])
    auc=roc_auc_score(y,s)
    rng=np.random.default_rng(42); bs=[]
    for _ in range(2000):
        i=rng.integers(0,len(s),len(s))
        if len(set(y[i]))<2: continue
        bs.append(roc_auc_score(y[i],s[i]))
    ci=[float(np.percentile(bs,2.5)),float(np.percentile(bs,97.5))]
    obs=auc; cnt=0
    for _ in range(20000):
        if roc_auc_score(y[rng.permutation(len(y))],s)>=obs-1e-12: cnt+=1
    p=(cnt+1)/20001
    out={'model':name.upper(),'gate':'FAILED (effective rank <10; benchmark for transparency only)',
      'trial_AUROC':float(auc),'CI95':ci,'permutation_p':p,'n_evaluated':len(ev),
      'n_success':int(y.sum()),'n_failure':int((1-y).sum()),
      'unique_scores':int(len(set(np.round(s,6)))),'per_drug':rows}
    json.dump(out,open(f'results/trial_benchmark_{name}.json','w'),indent=1)
    print(name.upper(),'AUROC %.3f CI %.2f-%.2f perm p=%.4f unique scores %d/%d'%(auc,*ci,p,out['unique_scores'],len(ev)))
