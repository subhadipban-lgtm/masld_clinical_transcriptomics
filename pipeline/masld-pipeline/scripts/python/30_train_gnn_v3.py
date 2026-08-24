"""Task 4: retrain GraphSAGE on KG v3, 5 seeds, 200 epochs w/ early stopping.
Node features: genes -> signature logFC/padj/universe; drugs -> zeros (no SMILES/ATC
available in merged table; Methods limitation). Negatives: random pairs excluded from
ALL observed edges AND from the curated drug-target union (CTD+DrugBank+DrugCentral) -
this implements the filtering the manuscript previously claimed but never had."""
import sys, json, statistics
sys.path.insert(0,'masld-cdss/pipeline/masld-pipeline/scripts/python')
import numpy as np, pandas as pd, networkx as nx, torch
from masldgnn.model import GraphSAGE_LinkPredictor
from sklearn.metrics import roc_auc_score, average_precision_score
import torch.nn as nn

torch.manual_seed(0)
G=nx.read_gexf('data/kg/masld_kg_v3.gexf')
sig=pd.read_csv('results/ws1_signature/ws1_signature_genes.csv').set_index('GeneSymbol')
univ=set(pd.read_pickle('results/decisive_test/discovery_qnorm.pkl').index)
nodes=sorted(G.nodes()); idx={n:i for i,n in enumerate(nodes)}
types=np.array([1 if G.nodes[n].get('type')=='gene' else 0 for n in nodes])
lf=np.array([sig.logFC.get(n,0.0) if n in sig.index else 0.0 for n in nodes])
pv=np.array([-np.log10(sig['adj.P.Val'].get(n,1.0)) if n in sig.index else 0.0 for n in nodes])
un=np.array([1.0 if n in univ else 0.0 for n in nodes])
X=torch.tensor(np.vstack([types,lf,pv,un]).T,dtype=torch.float)
edges=[(u,v) for u,v in G.edges()]
E=torch.tensor([[idx[u],idx[v]] for u,v in edges]+[[idx[v],idx[u]] for u,v in edges],dtype=torch.long).t()
# curated known pairs for negative filtering
dt=pd.read_csv('data/kg/drug_target_merged.csv')
known={(a,b) for a,b in zip(dt.drug_name.str.lower(),dt.target_gene)}
known|={(b,a) for a,b in known}
drugnodes={n for n in nodes if G.nodes[n].get('type')=='drug'}
genenodes={n for n in nodes if G.nodes[n].get('type')=='gene'}
observed={(u,v) for u,v in edges}|{(v,u) for u,v in edges}
rng=np.random.default_rng(42)
def sample_neg(n):
    out=[]; dl=list(drugnodes); gl=list(genenodes)
    while len(out)<n:
        a=dl[rng.integers(len(dl))]; b=gl[rng.integers(len(gl))]
        if (a,b) not in observed and (a.lower(),b) not in known and (b,a) not in known:
            out.append((a,b))
    return out
pos=rng.permutation(len(edges))
n_tr,n_va=int(.8*len(edges)),int(.1*len(edges))
tr=[edges[i] for i in pos[:n_tr]]; va=[edges[i] for i in pos[n_tr:n_tr+n_va]]; te=[edges[i] for i in pos[n_tr+n_va:]]
neg_va=sample_neg(len(va)); neg_te=sample_neg(len(te))
def edge_t(pairs): return torch.tensor([[idx[a],idx[b]] for a,b in pairs],dtype=torch.long).t()
tr_t=edge_t(tr)
device='mps' if torch.backends.mps.is_available() else 'cpu'
def run(seed):
    torch.manual_seed(seed); np.random.seed(seed)
    neg_tr=sample_neg(len(tr))
    lab=torch.cat([torch.ones(len(tr)),torch.zeros(len(neg_tr))]).to(device)
    ei=torch.cat([tr_t,edge_t(neg_tr)],dim=1).to(device)
    train_ei=E.to(device)
    model=GraphSAGE_LinkPredictor(X.shape[1],64,32).to(device)
    opt=torch.optim.Adam(model.parameters(),lr=0.005)
    Xt=X.to(device); best=1e9; bad=0; 
    for ep in range(200):
        model.train(); opt.zero_grad()
        out=model.decode(model.encode(Xt,train_ei),ei)
        loss=nn.BCEWithLogitsLoss()(out,lab); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            zv=model.encode(Xt,train_ei)
            ov=model.decode(zv,edge_t(va).to(device)); vn=model.decode(zv,edge_t(neg_va).to(device))
            vl=nn.BCEWithLogitsLoss()(torch.cat([ov,vn]),torch.cat([torch.ones(len(va)),torch.zeros(len(neg_va))]).to(device)).item()
        if vl<best-1e-4: best=vl; bad=0; best_state={k:v.clone() for k,v in model.state_dict().items()}
        else:
            bad+=1
            if bad>=15: break
    model.load_state_dict(best_state); model.eval()
    with torch.no_grad():
        z=model.encode(Xt,train_ei)
        pv_=model.decode(z,edge_t(te).to(device)).cpu().numpy(); nv=model.decode(z,edge_t(neg_te).to(device)).cpu().numpy()
        y=np.r_[np.ones(len(pv_)),np.zeros(len(nv))]; s=np.r_[pv_,nv]
        auc=roc_auc_score(y,s); ap=average_precision_score(y,s)
        zn=z.cpu().numpy()
    return auc,ap,ep,zn
res=[]; Z=[]
for sd in range(5):
    auc,ap,ep,zn=run(sd); res.append((auc,ap,ep)); Z.append(zn)
    print(f'seed {sd}: AUROC={auc:.3f} AUPRC={ap:.3f} epochs={ep}')
a=[r[0] for r in res]; p=[r[1] for r in res]
print(f'AUROC {np.mean(a):.3f} ± {np.std(a):.3f} | AUPRC {np.mean(p):.3f} ± {np.std(p):.3f}')
# embedding diagnostics (drug nodes)
di=[]
for zn in Z:
    dz=zn[types==0]
    dz=dz/np.linalg.norm(dz,axis=1,keepdims=True)
    S=dz@dz.T
    iu=np.triu_indices(len(dz),1)
    ev=np.linalg.eigvalsh(np.cov(dz.T))[::-1]
    er=float((ev.sum())**2/ (ev**2).sum())
    di.append({'effective_rank':er,'mean_cosine':float(S[iu].mean()),'frac_gt099':float((S[iu]>0.99).mean())})
print(json.dumps(di,indent=1))
np.save('results/kg_v3_embeddings.npy',np.mean(Z,axis=0))
json.dump({'auroc':[float(x) for x in a],'auroc_mean':float(np.mean(a)),'auroc_sd':float(np.std(a)),
 'auprc_mean':float(np.mean(p)),'auprc_sd':float(np.std(p)),
 'epochs':[r[2] for r in res],'diagnostics':di,'n_nodes':len(nodes),'n_edges':G.number_of_edges()},
 open('results/gnn_v3_training.json','w'),indent=1)
