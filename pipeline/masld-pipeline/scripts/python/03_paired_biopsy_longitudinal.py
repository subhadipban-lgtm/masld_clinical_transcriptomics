"""WS2: longitudinal analysis of the locked 649-gene signature in the 58 paired Fujiwara biopsies."""
import json, numpy as np, pandas as pd
from pathlib import Path
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut, cross_val_predict

ROOT=Path("/Users/subhadipbanerjee/masld-revision"); KAM=ROOT/"data/kamzolas"
sig=pd.read_csv(ROOT/"results/ws1_signature/ws1_signature_genes.csv")
up=sig[sig.logFC>0].set_index("GeneSymbol").logFC; dn=sig[sig.logFC<0].set_index("GeneSymbol").logFC

C=pd.read_csv(KAM/"Fujiwara_dataset/raw_counts_fujiwara.csv", index_col=0)
cpm=np.log2(C/C.sum()*1e6+1); cpm=cpm[cpm.std(axis=1)>0]
etab=pd.read_csv(KAM/"ensembl_mapping.tsv",sep="\t"); sym2ens=dict(zip(etab.external_gene_name.astype(str).str.strip(), etab.ensembl_gene_id.astype(str).str.strip()))
u_ids=[sym2ens[g] for g in up.index if g in sym2ens and sym2ens[g] in cpm.index]
d_ids=[sym2ens[g] for g in dn.index if g in sym2ens and sym2ens[g] in cpm.index]
print("signature genes measured:",len(u_ids),"+",len(d_ids))
z=cpm.sub(cpm.mean(axis=1),axis=0).div(cpm.std(axis=1).replace(0,np.nan),axis=0)
score=(z.loc[u_ids].mul(up.reindex([s for s in up.index if sym2ens.get(s) in u_ids]).values,axis=0).sum(axis=0)
       + z.loc[d_ids].mul(dn.reindex([s for s in dn.index if sym2ens.get(s) in d_ids]).values,axis=0).sum(axis=0))
sc=pd.DataFrame({"score":score})
sc["patient"]=[i.rsplit("_",1)[0] for i in sc.index]; sc["biopsy"]=[i.rsplit("_",1)[1] for i in sc.index]
W=sc.pivot(index="patient",columns="biopsy")["score"].dropna()
print("paired patients:",len(W))
xl=pd.ExcelFile(KAM/"Fujiwara_dataset/Both biopsies - refined dataset.xlsx")
b1=xl.parse("1st_biopsy").set_index("Patient"); b2=xl.parse("2nd_biopsy").set_index("Patient")
pat=[p for p in W.index if p in b1.index and p in b2.index]
W=W.loc[pat]
dF=(b2.loc[pat,"Histology.fibrosis_2"].astype(float)-b1.loc[pat,"Histology.fibrosis"].astype(float))
dNAS=(b2.loc[pat,"NAS_2"].astype(float)-b1.loc[pat,"Histology.NAS"].astype(float))
base_stage=b1.loc[pat,"Histology.fibrosis"].astype(float)
delta=W.iloc[:,1]-W.iloc[:,0]  # biopsy 2 - biopsy 1 (columns sorted '1','2')
r1=stats.spearmanr(delta,dF); r2=stats.spearmanr(delta,dNAS)
prog=(dF>0).astype(int)
X=pd.DataFrame({"score":W.iloc[:,0],"base_stage":base_stage.values})
y=prog.values
pidx=np.arange(len(y))
loo=LeaveOneOut()
pred=cross_val_predict(LogisticRegression(max_iter=2000),X,y,cv=loo,method="predict_proba")[:,1]
from sklearn.metrics import roc_auc_score
auc=roc_auc_score(y,pred)
# exact binomial CI
lo,hi=stats.beta.ppf(0.025,max(sum((pred>=.5)==(y==1)),1),len(y)-sum((pred>=.5)==(y==1))+1), \
      stats.beta.ppf(0.975,sum((pred>=.5)==(y==1))+1,len(y)-sum((pred>=.5)==(y==1)))
out={"n_paired":int(len(W)),"n_signature_genes_measured":int(len(u_ids)+len(d_ids)),
 "delta_score_vs_delta_fibrosis":{"rho":float(r1.statistic),"p":float(r1.pvalue)},
 "delta_score_vs_delta_NAS":{"rho":float(r2.statistic),"p":float(r2.pvalue)},
 "progressed":int(y.sum()),"regressed":int((dF<0).sum()),"stable":int((dF==0).sum()),
 "prognostic_LOOCV_baseline_score_adj_baseline_stage":{"AUROC":float(auc),
   "acc":float(((pred>=.5)==y).mean()),"binom95CI":[float(lo),float(hi)],"n":int(len(y))}}
print(json.dumps(out,indent=1))
Wf=pd.DataFrame({"baseline_score":W.iloc[:,0],"followup_score":W.iloc[:,1],"delta_score":delta,
  "baseline_stage":base_stage,"delta_fibrosis":dF,"delta_NAS":dNAS,"progressed":y},index=pat)
Wf.to_csv(ROOT/"results/ws1_signature/ws2_paired_signature_scores.csv")
json.dump(out,open(ROOT/"results/ws1_signature/stats_ws2_longitudinal.json","w"),indent=1)
