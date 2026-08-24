"""Bulk deconvolution of the 75-gene panel scores via NNLS against two liver
single-cell references (Tabula Sapiens v2; MacParland GSE115469).
Derived inputs built from locked artifacts: bulk = results/modules qnorm matrices;
scores = 75-gene directional score; metadata = stages + discovery cohort file."""
import os, json, pickle
import numpy as np, pandas as pd
import anndata as ad
from scipy.optimize import nnls
from scipy import stats as st
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

OUT='output'; os.makedirs(OUT,exist_ok=True)
mats=pickle.load(open('results/modules/qnorm_matrices.pkl','rb'))
stages=pickle.load(open('results/modules/stages.pkl','rb'))
d349=pd.read_csv('data/discovery_cohort_349.csv').set_index('sample_id')
full=pd.read_csv('results/ws1_signature/ws1_dge_full.csv').set_index('GeneSymbol')
p75=pd.read_csv('results/ws11/panel_size75_genes.csv')
UP=[g for g in p75.GeneSymbol if full.loc[g,'logFC']>0]
DN=[g for g in p75.GeneSymbol if full.loc[g,'logFC']<0]

bulk={}; meta_rows=[]
for c,M in mats.items():
    if c=='Discovery':
        stage=d349.loc[M.columns,'fibrosis_stage']
    else:
        stage=stages[c].reindex(M.columns)
    M=M.loc[:,stage.notna()]; stage=stage[stage.notna()]
    z=M.sub(M.mean(axis=1),axis=0).div(M.std(axis=1,ddof=1),axis=0).fillna(0)
    sc=z.loc[[g for g in UP if g in z.index]].mean(axis=0)-z.loc[[g for g in DN if g in z.index]].mean(axis=0)
    bulk[c]={'M':M,'stage':stage,'score':sc}
    for s,v in sc.items(): meta_rows.append({'sample':f'{c}:{s}','cohort':c,'stage':stage[s],'score':v})
meta=pd.DataFrame(meta_rows)
meta['stage_factor']=meta.stage.map(lambda x:f'F{int(x)}')
meta.to_csv(f'{OUT}/metadata.csv',index=False)
p75.to_csv(f'{OUT}/panel_75_genes.csv',index=False)

BROAD={
 'Hepatocytes':{'hepatocyte'},
 'Stellate':{'hepatic stellate cell'},
 'Macrophages':{'macrophage','monocyte','intermediate monocyte','classical monocyte','non-classical monocyte'},
 'Endothelial':{'endothelial cell'},
 'Cholangiocytes':{'intrahepatic cholangiocyte'},
 'T_cells':{'cd8-positive, alpha-beta t cell','cd4-positive, alpha-beta t cell','t cell'},
 'NK_cells':{'natural killer cell','mature nk t cell'},
 'B_cells':{'b cell','plasma cell'}}

def build_tsp(min_cells=50,chunk=400):
    A=ad.read_h5ad('data/Liver_TSP1_30_version2d_10X_smartseq_scvi_Nov122024.h5ad',backed='r')
    genes=np.array(A.var_names,dtype=object)
    occ=A.obs.cell_ontology_class.astype(str).values
    groups={b:np.where(np.isin(occ,list(mems)))[0] for b,mems in BROAD.items()}
    sig={}
    for b,idx in groups.items():
        if len(idx)<min_cells: 
            print(f'TSP {b}: {len(idx)} cells < {min_cells}, EXCLUDED'); continue
        acc=np.zeros(len(genes),dtype=np.float64); n=0
        for i in range(0,len(idx),chunk):
            sub=idx[i:i+chunk]
            X=A.layers['raw_counts'][sub].toarray().astype(np.float64)
            rs=X.sum(axis=1,keepdims=True); rs[rs==0]=1
            acc+=np.log2(X/rs*1e6+1).sum(axis=0); n+=len(sub)
        sig[b]=acc/n
        print(f'TSP {b}: {n} cells')
    A.file.close()
    S=pd.DataFrame(sig,index=genes)
    return S

MAC_MAP=[('Hepatocytes',('Hepatocyte_',)),('Macrophages',('Inflammatory_Macrophage','Non-inflammatory_Macrophage')),
 ('Endothelial',('Periportal_LSECs','Central_venous_LSECs','Portal_endothelial_Cells')),
 ('Cholangiocytes',('Cholangiocytes',)),('T_cells',('alpha-beta_T_Cells','gamma-delta_T_Cells')),
 ('NK_cells',('NK-like_Cells',)),('B_cells',('Mature_B_Cells','Plasma_Cells')),
 ('Stellate',('Hepatic_Stellate_Cells',))]
def build_mac(min_cells=50):
    expr=pd.read_csv('data/GSE115469_Data.csv.gz',index_col=0).T  # genes x cells -> cells x genes
    lab=pd.read_csv('data/GSE115469_CellClusterType.txt.gz',sep='\t').set_index('CellName')
    ct=lab.reindex(expr.index).CellType.astype(str)
    rs=expr.sum(axis=1); rs[rs==0]=1
    lcp=np.log2(expr.div(rs,axis=0)*1e6+1)
    sig={}
    for b,prefixes in MAC_MAP:
        m=ct.str.startswith(prefixes)
        if m.sum()<min_cells:
            print(f'MAC {b}: {m.sum()} cells < {min_cells}, EXCLUDED'); continue
        sig[b]=lcp[m].mean(axis=0).values
        print(f'MAC {b}: {int(m.sum())} cells')
    return pd.DataFrame(sig,index=expr.columns.astype(str)),lcp,m.sum() if False else None

def deconvolve(sig,bulk_mats):
    allfr=[]
    for c,d in bulk_mats.items():
        common=[g for g in sig.index if g in d['M'].index]
        S=sig.loc[common].values
        B=d['M'].loc[common].values
        fr=np.array([nnls(S,B[:,j])[0] for j in range(B.shape[1])])
        fr=fr/fr.sum(axis=1,keepdims=True)
        f=pd.DataFrame(fr,index=d['M'].columns,columns=sig.columns)
        f['sample']=[f'{c}:{x}' for x in f.index]; f['cohort']=c
        allfr.append(f)
    return pd.concat(allfr,ignore_index=True)

def corr_tables(fr,meta):
    m=meta.merge(fr.drop(columns=['cohort']),on='sample')
    rows=[];rows2=[]
    for c in m.cohort.unique():
        mc=m[m.cohort==c]
        for ct_ in fr.columns:
            if ct_ in ('sample','cohort'): continue
            r=st.spearmanr(mc[ct_],mc.score)
            rows.append({'cohort':c,'cell_type':ct_,'rho':round(float(r.statistic),3),'p':float(r.pvalue),'n':int(len(mc))})
        for grp,lo,hi in [('F0-1',0,1),('F2-4',2,4)]:
            g=mc[(mc.stage>=lo)&(mc.stage<=hi)]
            if len(g)<10: continue
            for ct_ in fr.columns:
                if ct_ in ('sample','cohort'): continue
                r=st.spearmanr(g[ct_],g.score)
                rows2.append({'cohort':c,'stage_group':grp,'cell_type':ct_,'rho':round(float(r.statistic),3),'p':float(r.pvalue),'n':int(len(g))})
    return pd.DataFrame(rows),pd.DataFrame(rows2)

for name,builder in [('tabula_sapiens',lambda: build_tsp()),('macparland',lambda: build_mac()[0])]:
    S=builder()
    S.to_csv(f'{OUT}/signature_{name}.csv')
    fr=deconvolve(S,bulk)
    fr.to_csv(f'{OUT}/fractions_{name.split("_")[0] if name=="tabula_sapiens" else "macparland"}.csv',index=False)
    tag='tabula' if 'tabula' in name else 'macparland'
    c1,c2=corr_tables(fr,meta)
    c1.to_csv(f'{OUT}/correlation_{tag}.csv',index=False)
    c2.to_csv(f'{OUT}/correlation_by_stage_{tag}.csv',index=False)
    print(f'== {name}: fractions {fr.shape}, mean by type:')
    print(fr.drop(columns=['sample','cohort']).mean().round(3).to_string())
    print(f'== {name} stellate/cholangiocyte rho by cohort:')
    print(c1[c1.cell_type.isin(['Stellate','Cholangiocytes'])].to_string(index=False)) if len(c1) else print('(empty)')
print('DONE')
