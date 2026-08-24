"""Task 6: graph baselines on the identical link-prediction split (seed 42)."""
import sys, json, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0,'masld-cdss/pipeline/masld-pipeline/scripts/python')
import numpy as np, pandas as pd, networkx as nx, torch
import torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch_geometric.nn import GCNConv, GATConv
torch.manual_seed(0)
G=nx.read_gexf('data/kg/masld_kg_v3.gexf')
nodes=sorted(G.nodes()); idx={n:i for i,n in enumerate(nodes)}
edges=[(u,v) for u,v in G.edges()]
sig=pd.read_csv('results/ws1_signature/ws1_signature_genes.csv').set_index('GeneSymbol')
univ=set(pd.read_pickle('results/decisive_test/discovery_qnum.pkl').index) if False else set(pd.read_pickle('results/decisive_test/discovery_qnorm.pkl').index)
types=np.array([1 if G.nodes[n].get('type')=='gene' else 0 for n in nodes])
X=torch.tensor(np.vstack([types,[sig.logFC.get(n,0.0) if n in sig.index else 0.0 for n in nodes],
 [-np.log10(sig['adj.P.Val'].get(n,1.0)) if n in sig.index else 0.0 for n in nodes],
 [1.0 if n in univ else 0.0 for n in nodes]]).T,dtype=torch.float)
rng=np.random.default_rng(42)
pos=rng.permutation(len(edges)); n_tr,n_va=int(.8*len(edges)),int(.1*len(edges))
tr=[edges[i] for i in pos[:n_tr]]; va=[edges[i] for i in pos[n_tr:n_tr+n_va]]; te=[edges[i] for i in pos[n_tr+n_va:]]
drugnodes=[n for n in nodes if G.nodes[n].get('type')=='drug']; genenodes=[n for n in nodes if G.nodes[n].get('type')=='gene']
observed={(u,v) for u,v in edges}|{(v,u) for u,v in edges}
dt=pd.read_csv('data/kg/drug_target_merged.csv')
known={(a.lower(),b) for a,b in zip(dt.drug_name,dt.target_gene)}
def neg(m):
    out=[]
    while len(out)<m:
        a=drugnodes[rng.integers(len(drugnodes))]; b=genenodes[rng.integers(len(genenodes))]
        if (a,b) not in observed and (a.lower(),b) not in known: out.append((a,b))
    return out
neg_te=neg(len(te))
y=np.r_[np.ones(len(te)),np.zeros(len(neg_te))]
A=set(map(tuple,te))
def cn_score(pairs):
    Gtr=nx.Graph(); Gtr.add_nodes_from(nodes); Gtr.add_edges_from(tr)
    return np.array([len(set(Gtr.neighbors(a))&set(Gtr.neighbors(b))) for a,b in pairs],dtype=float)
sc=cn_score(te+neg_te); print('common-neighbour AUROC %.3f'%roc_auc_score(y,sc))
E=torch.tensor([[idx[a],idx[b]] for a,b in edges]+[[idx[b],idx[a]] for a,b in edges],dtype=torch.long).t()
def tt(pairs): return torch.tensor([[idx[a],idx[b]] for a,b in pairs],dtype=torch.long).t()
tr_t=tt(tr); te_t=tt(te+neg_te)
def run_gnn(make_model, nfeat=None, emb_fn=None, seeds=5):
    aucs=[]
    for sd in range(seeds):
        torch.manual_seed(sd)
        if emb_fn is not None:  # node2vec
            emb=emb_fn(sd)
        else:
            m=make_model(); opt=torch.optim.Adam(m.parameters(),lr=0.01)
            lab=torch.cat([torch.ones(len(tr)),torch.zeros(len(neg(sd and 0 or 0)))] ) # placeholder
            for ep in range(100):
                m.train(); opt.zero_grad()
                ng=neg(len(tr))
                ei=torch.cat([tr_t,tt(ng)],dim=1)
                lb=torch.cat([torch.ones(len(tr)),torch.zeros(len(ng))])
                out=m(X, E, ei); loss=F.binary_cross_entropy_with_logits(out,lb); loss.backward(); opt.step()
            m.eval()
            with torch.no_grad(): s=m(X,E,te_t).numpy()
            aucs.append(roc_auc_score(y,s)); continue
        s=emb[te_t[0]]*emb[te_t[1]]
        from sklearn.linear_model import LogisticRegression
        ntr=neg(len(tr)); nt_t=tt(ntr)
        Xall=torch.cat([tr_t,nt_t],dim=1)
        Xtr=np.c_[emb[Xall[0]].numpy(),emb[Xall[1]].numpy()]; ytr=np.r_[np.ones(len(tr)),np.zeros(len(ntr))]
        Xte=np.c_[emb[te_t[0]].numpy(),emb[te_t[1]].numpy()]
        lr=LogisticRegression(max_iter=1000).fit(Xtr,ytr)
        aucs.append(roc_auc_score(y,lr.predict_proba(Xte)[:,1]))
    return aucs
def n2v(sd, dim=64, walklen=80, walks=10, win=10, epochs=5):
    # pure-python deepwalk-style node2vec fallback (torch-cluster unavailable)
    import random
    random.seed(sd); np.random.seed(sd)
    adj={i:set() for i in range(len(nodes))}
    for a,b in tr:
        adj[idx[a]].add(idx[b]); adj[idx[b]].add(idx[a])
    walksL=[]
    for _ in range(walks):
        for start in adj:
            w=[start]; cur=start
            for _ in range(walklen):
                nb=adj[cur]
                if not nb: break
                cur=random.choice(list(nb)); w.append(cur)
            walksL.append(w)
    from gensim.models import Word2Vec
    m=Word2Vec([list(map(str,w)) for w in walksL], vector_size=dim, window=win, min_count=0,
               sg=1, workers=4, epochs=epochs, seed=sd)
    emb=torch.tensor(np.vstack([m.wv[str(i)] for i in range(len(nodes))]))
    return emb
class GCN(nn.Module):
    def __init__(self):
        super().__init__(); self.c1=GCNConv(4,64); self.c2=GCNConv(64,64); self.l=nn.Linear(128,1)
    def forward(self,x,ei,eidx):
        h=F.relu(self.c1(x,ei)); h=self.c2(h,ei)
        return self.l(torch.cat([h[eidx[0]],h[eidx[1]]],dim=1)).squeeze(1)
class GAT(nn.Module):
    def __init__(self):
        super().__init__(); self.c1=GATConv(4,16,heads=4); self.c2=GATConv(64,16,heads=4); self.l=nn.Linear(128,1)
    def forward(self,x,ei,eidx):
        h=F.elu(self.c1(x,ei)); h=self.c2(h,ei)
        return self.l(torch.cat([h[eidx[0]],h[eidx[1]]],dim=1)).squeeze(1)
res={}
res['common_neighbour']=float(roc_auc_score(y,sc))
res['node2vec']=[float(a) for a in run_gnn(None,emb_fn=n2v)]
res['GCN']=[float(a) for a in run_gnn(GCN)]
res['GAT']=[float(a) for a in run_gnn(GAT)]
for k,v in res.items():
    if isinstance(v,list): print(f'{k}: {np.mean(v):.3f} ± {np.std(v):.3f}')
    else: print(f'{k}: {v:.3f}')
json.dump(res,open('results/baselines.json','w'),indent=1)
