"""WS4/WS5 re-run: GCN and GAT as primary models. Identical protocol to GraphSAGE run
(script 30): same KG v3, same 80/10/10 split (seed 42), same filtered negatives
(CTD+DrugBank+DrugCentral), early stopping patience 15, seeds 0-4, 32-dim node
embeddings decoded by dot product."""
import sys, json
sys.path.insert(0,'masld-cdss/pipeline/masld-pipeline/scripts/python')
import numpy as np, pandas as pd, networkx as nx, torch, torch.nn as nn, torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
from sklearn.metrics import roc_auc_score, average_precision_score

class GCNEnc(nn.Module):
    def __init__(self):
        super().__init__(); self.c1=GCNConv(4,64); self.c2=GCNConv(64,32)
    def encode(self,x,ei): return self.c2(F.relu(self.c1(x,ei)),ei)
    def decode(self,z,e): return (z[e[0]]*z[e[1]]).sum(dim=1)

class GATEnc(nn.Module):
    def __init__(self):
        super().__init__(); self.c1=GATConv(4,16,heads=4); self.c2=GATConv(64,8,heads=4)
    def encode(self,x,ei): return self.c2(F.elu(self.c1(x,ei)),ei)
    def decode(self,z,e): return (z[e[0]]*z[e[1]]).sum(dim=1)

torch.manual_seed(0)
G=nx.read_gexf('data/kg/masld_kg_v3.gexf')
sig=pd.read_csv('results/ws1_signature/ws1_signature_genes.csv').set_index('GeneSymbol')
univ=set(pd.read_pickle('results/decisive_test/discovery_qnorm.pkl').index)
nodes=sorted(G.nodes()); idx={n:i for i,n in enumerate(nodes)}
types=np.array([1 if G.nodes[n].get('type')=='gene' else 0 for n in nodes])
X=torch.tensor(np.vstack([types,
 [sig.logFC.get(n,0.0) if n in sig.index else 0.0 for n in nodes],
 [-np.log10(sig['adj.P.Val'].get(n,1.0)) if n in sig.index else 0.0 for n in nodes],
 [1.0 if n in univ else 0.0 for n in nodes]]).T,dtype=torch.float)
edges=[(u,v) for u,v in G.edges()]
E=torch.tensor([[idx[u],idx[v]] for u,v in edges]+[[idx[v],idx[u]] for u,v in edges],dtype=torch.long).t()
dt=pd.read_csv('data/kg/drug_target_merged.csv')
known={(a.lower(),b) for a,b in zip(dt.drug_name,dt.target_gene)}
drugnodes=[n for n in nodes if G.nodes[n].get('type')=='drug']; genenodes=[n for n in nodes if G.nodes[n].get('type')=='gene']
observed={(u,v) for u,v in edges}|{(v,u) for u,v in edges}
rng=np.random.default_rng(42)
def sample_neg(n):
    out=[]
    while len(out)<n:
        a=drugnodes[rng.integers(len(drugnodes))]; b=genenodes[rng.integers(len(genenodes))]
        if (a,b) not in observed and (a.lower(),b) not in known: out.append((a,b))
    return out
pos=rng.permutation(len(edges)); n_tr,n_va=int(.8*len(edges)),int(.1*len(edges))
tr=[edges[i] for i in pos[:n_tr]]; va=[edges[i] for i in pos[n_tr:n_tr+n_va]]; te=[edges[i] for i in pos[n_tr+n_va:]]
neg_va=sample_neg(len(va)); neg_te=sample_neg(len(te))
def et(p): return torch.tensor([[idx[a],idx[b]] for a,b in p],dtype=torch.long).t()
tr_t=et(tr); va_t=et(va); nv_t=et(neg_va); te_t=et(te); nt_t=et(neg_te)
dev='cpu'
def train_model(Maker, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    neg_tr=sample_neg(len(tr))
    lab=torch.cat([torch.ones(len(tr)),torch.zeros(len(neg_tr))])
    ei=torch.cat([tr_t,et(neg_tr)],dim=1)
    m=Maker(); opt=torch.optim.Adam(m.parameters(),lr=0.005)
    yv=torch.cat([torch.ones(len(va)),torch.zeros(len(neg_va))])
    best=1e9; bad=0; best_state=None
    for ep in range(200):
        m.train(); opt.zero_grad()
        loss=nn.BCEWithLogitsLoss()(m.decode(m.encode(X,E),ei),lab); loss.backward(); opt.step()
        m.eval()
        with torch.no_grad():
            z=m.encode(X,E)
            vl=nn.BCEWithLogitsLoss()(torch.cat([m.decode(z,va_t),m.decode(z,nv_t)]),yv).item()
        if vl<best-1e-4: best=vl; bad=0; best_state={k:v.clone() for k,v in m.state_dict().items()}
        else:
            bad+=1
            if bad>=15: break
    m.load_state_dict(best_state); m.eval()
    with torch.no_grad():
        z=m.encode(X,E).numpy()
        ps=m.decode(m.encode(X,E),te_t).numpy(); ns=m.decode(m.encode(X,E),nt_t).numpy()
    y=np.r_[np.ones(len(te)),np.zeros(len(neg_te))]; s=np.r_[ps,ns]
    auc=roc_auc_score(y,s); ap=average_precision_score(y,s)
    dz=z[types==0]; dz=dz/np.linalg.norm(dz,axis=1,keepdims=True)
    S=dz@dz.T; iu=np.triu_indices(len(dz),1)
    ev=np.linalg.eigvalsh(np.cov(dz.T))[::-1]
    er=float(ev.sum()**2/(ev**2).sum())
    diag={'effective_rank':er,'mean_cosine':float(S[iu].mean()),'frac_gt099':float((S[iu]>0.99).mean())}
    return auc,ap,ep,diag,z

for name,maker in [('gcn',GCNEnc),('gat',GATEnc)]:
    aucs=[];aps=[];diags=[];Zs=[]
    for sd in range(5):
        auc,ap,ep,di,z=train_model(maker,sd); aucs.append(auc);aps.append(ap);diags.append(di);Zs.append(z)
        print(f'{name} seed {sd}: AUROC={auc:.3f} AUPRC={ap:.3f} ep={ep} ER={di["effective_rank"]:.1f} cos={di["mean_cosine"]:.3f} f99={di["frac_gt099"]:.3f}')
    out={'auroc':[float(a) for a in aucs],'auroc_mean':float(np.mean(aucs)),'auroc_sd':float(np.std(aucs)),
         'auprc_mean':float(np.mean(aps)),'auprc_sd':float(np.std(aps)),
         'diagnostics':diags,'n_nodes':len(nodes),'n_edges':G.number_of_edges(),
         'diagnostics_mean':{k:float(np.mean([d[k] for d in diags])) for k in diags[0]}}
    json.dump(out,open(f'results/{name}_training.json','w'),indent=1)
    np.save(f'results/{name}_embeddings.npy',np.mean(Zs,axis=0))
    print(name,'MEAN: AUROC %.3f±%.3f  ER %.2f  cos %.3f  f99 %.3f'%(np.mean(aucs),np.std(aucs),
        out['diagnostics_mean']['effective_rank'],out['diagnostics_mean']['mean_cosine'],out['diagnostics_mean']['frac_gt099']))
