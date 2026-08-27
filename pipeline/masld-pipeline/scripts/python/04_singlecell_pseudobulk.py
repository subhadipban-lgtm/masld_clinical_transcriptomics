"""WS3: signature score per donor x cell type (pseudobulk, GSE136103 Ramachandran).
Test across donors: cirrhotic vs healthy within each cell type (donors are replicates)."""
import json, numpy as np, pandas as pd
from scipy import stats
pb=pd.read_csv("MASLD/scRNAanalysis/Pseudobulk_Donor_CellType.csv").set_index("Gene")
sig=pd.read_csv("results/ws1_signature/ws1_signature_genes.csv")
up=dict(zip(sig[sig.logFC>0].GeneSymbol,sig[sig.logFC>0].logFC)); dn=dict(zip(sig[sig.logFC<0].GeneSymbol,sig[sig.logFC<0].logFC))
common=lambda dd: [g for g in dd if g in pb.index]
u,d=common(up),common(dn); print("genes measured:",len(u),"+",len(d))
cpm=pb.div(pb.sum(axis=0)/1e6,axis=1)
z=cpm.sub(cpm.mean(axis=1),axis=0).div(cpm.std(axis=1).replace(0,np.nan),axis=0).fillna(0)
score=(z.loc[u].mul(pd.Series({g:up[g] for g in u}),axis=0).sum(axis=0)
      +z.loc[d].mul(pd.Series({g:dn[g] for g in d}),axis=0).sum(axis=0))
df=pd.DataFrame({"sample":score.index,"score":score.values})
df["donor"]=df["sample"].str.split("-",n=1).str[0]; df["cell_type"]=df["sample"].str.split("-",n=1).str[1]
piv=df.pivot(index="donor",columns="cell_type",values="score")
grp=piv.index.str.contains("cirrhotic")
out={"n_donors":int(piv.shape[0]),"n_cirrhotic":int(grp.sum()),"n_healthy":int((~grp).sum()),
     "genes_measured":int(len(u)+len(d)),"cell_types":{}}
for ct in piv.columns:
    a=piv.loc[grp,ct].dropna(); b=piv.loc[~grp,ct].dropna()
    if len(a)>=3 and len(b)>=3:
        r=stats.mannwhitneyu(a,b)
        out["cell_types"][ct]={"mean_cirrhotic":float(a.mean()),"mean_healthy":float(b.mean()),
            "U_p":float(r.pvalue),"n_c":int(len(a)),"n_h":int(len(b))}
print(json.dumps(out,indent=1))
json.dump(out,open("results/ws1_signature/stats_ws3_singlecell.json","w"),indent=1)
df.to_csv("results/ws1_signature/ws3_pseudobulk_signature_scores.csv")
