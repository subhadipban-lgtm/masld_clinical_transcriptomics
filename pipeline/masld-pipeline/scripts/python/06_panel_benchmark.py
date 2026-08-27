"""WS6: comparative benchmark of the 649-gene signature vs published panels.
Locked protocol: same normalisation as WS1; seed 42; 2000 bootstrap CIs."""
import json, re, numpy as np, pandas as pd
from pathlib import Path
from scipy import stats as st
from sklearn.metrics import roc_auc_score

ROOT=Path('.'); OUT=ROOT/'results/ws6_benchmark'; rng=np.random.default_rng(42)
NB=2000
# ---- panels ----
sig=pd.read_csv('results/ws1_signature/ws1_signature_genes.csv')
panels={'Ours_649':(sig[sig.logFC>0].GeneSymbol.tolist(), sig[sig.logFC<0].GeneSymbol.tolist(),True)}
p145=pd.read_csv('data/kamzolas/145 (from200) most important genes (variable importance with MeanDecreaseAccuracy>0).csv')
panels['Kamzolas_145']=(p145.SYMBOL.astype(str).str.strip().tolist(),[],False)
p57=pd.read_csv('data/kamzolas/57_BMs.csv')
panels['Kamzolas_57BM']=(p57.external_gene_name.astype(str).str.strip().tolist(),[],False)
p15=pd.read_csv('data/kamzolas/top_15_BMs_from_ELBOW.csv')
panels['Kamzolas_15BM']=(p15.external_gene_name.astype(str).str.strip().tolist(),[],False)
p194=[l.strip() for l in open('data/kamzolas/194_proteo_transcriptomic_signature.txt') if l.strip()]
p194=[re.split(r'[;,]',g)[0].strip() for g in p194]
panels['Kamzolas_194PT']=(p194,[],False)
p3=pd.read_csv('data/kamzolas/External_3_gene_panel_for_additional_validation.csv')
panels['Kamzolas_3gene']=(p3.external_gene_name.astype(str).str.strip().tolist(),[],False)
for k,(u,d,_) in panels.items(): print(k,'up',len(u),'down',len(d))

# ---- cohorts (WS1 qnorm matrices) ----
cohorts={}
D=pd.read_pickle('results/decisive_test/discovery_qnorm.pkl')
m349=pd.read_csv('data/discovery_cohort_349.csv').set_index('sample_id')
cohorts['Discovery']=(D,(m349.loc[D.columns,'fibrosis_group']=='Late').astype(int).values,m349.loc[D.columns,'fibrosis_stage'].values)
fm=pd.read_csv('data/metadata_with_ferroptosis_scores.csv'); fm['col']=fm['dataset']+'.'+fm['title']; fm=fm.set_index('col')
F=pd.read_pickle('results/decisive_test/fujiwara_qnorm.pkl')
fs=pd.to_numeric(fm['fibrosis stage:ch1'],errors='coerce').reindex(F.columns)
cohorts['Fujiwara']=(F.loc[:,fs.notna()],(fs[fs.notna()]>=3).astype(int).values,fs[fs.notna()].values)
E=pd.read_pickle('results/decisive_test/epos_qnorm.pkl')
em=pd.read_csv('data/kamzolas/EPoS_dataset/epos_metadata.csv').set_index('GEO_ID')
es=pd.to_numeric(em['Fibrosis.stage'],errors='coerce').reindex(E.columns)
cohorts['EPoS']=(E.loc[:,es.notna()],(es[es.notna()]>=3).astype(int).values,es[es.notna()].values)
U=pd.read_pickle('results/decisive_test/ucam_qnorm.pkl')
um=pd.read_csv('data/kamzolas/ucam_sanyal/metadata.csv').set_index('Sample name')
us=pd.to_numeric(um['Fibrosis'],errors='coerce').reindex(U.columns)
cohorts['UCAM_Sanyal']=(U.loc[:,us.notna()],(us[us.notna()]>=3).astype(int).values,us[us.notna()].values)

def zs(df): return df.sub(df.mean(axis=1),axis=0).div(df.std(axis=1,ddof=1),axis=0).fillna(0)
def boot_auc(y,s):
    y=np.asarray(y); s=np.asarray(s); a=roc_auc_score(y,s); bs=[]
    for _ in range(NB):
        i=rng.integers(0,len(s),len(s))
        if len(set(y[i]))<2: continue
        bs.append(roc_auc_score(y[i],s[i]))
    return a,float(np.percentile(bs,2.5)),float(np.percentile(bs,97.5))
def score_panel(zdf,up,dn,directional):
    u=[g for g in up if g in zdf.index]; d=[g for g in dn if g in zdf.index]
    if directional and d: return zdf.loc[u].mean(axis=0)-zdf.loc[d].mean(axis=0)
    return zdf.loc[u].mean(axis=0)

def delong_test(y,sa,sb):
    # DeLong covariance-based test for two correlated AUCs (same samples)
    y=np.asarray(y,int); sa=np.asarray(sa,float); sb=np.asarray(sb,float)
    n1=int(y.sum()); n0=len(y)-n1
    if n1==0 or n0==0: return np.nan,np.nan,np.nan
    def midrank(x):
        J=np.argsort(x); Z=x[J]; r=np.empty(len(x)); i=0
        while i<len(x):
            j=i
            while j+1<len(x) and Z[j+1]==Z[i]: j+=1
            r[J[i:j+1]]=(i+j+2)/2.0; i=j+1
        return r
    X1=sa[y==1]; X0=sa[y==0]; Y1=sb[y==1]; Y0=sb[y==0]
    ra=midrank(np.r_[X1,X0]); rb=midrank(np.r_[Y1,Y0])
    n=n1+n0
    def v10(r,m1,m0):
        return r[:m1]/n1 - (r[m1:]/n0).mean()
    def v01(r,m1,m0):
        return (r[:m1]/n1).mean() - r[m1:]/n0
    va10,va01=v10(ra,n1,n0),v01(ra,n1,n0)
    vb10,vb01=v10(rb,n1,n0),v01(rb,n1,n0)
    S10=np.cov(np.vstack([va10,vb10])); S01=np.cov(np.vstack([va01,vb01]))
    S=S10/n1+S01/n0
    diff=roc_auc_score(y,sa)-roc_auc_score(y,sb)
    var=S[0,0]+S[1,1]-2*S[0,1]
    if var<=0: return diff,np.nan,np.nan
    z=diff/np.sqrt(var); p=2*(1-st.norm.cdf(abs(z)))
    return diff, diff-1.96*np.sqrt(var), diff+1.96*np.sqrt(var), p

rows=[]; scores={}
for pname,(up,dn,directional) in panels.items():
    for cname,(df,y,stage) in cohorts.items():
        z=zs(df)
        cov=(len([g for g in up+dn if g in z.index]))/len(set(up+dn))
        sc=score_panel(z,up,dn,directional)
        a,lo,hi=boot_auc(y,sc)
        rho=st.spearmanr(sc,stage)
        rows.append({'panel':pname,'cohort':cname,'n':len(y),'coverage':round(cov,3),
            'AUROC':round(a,3),'CI_lo':round(lo,3),'CI_hi':round(hi,3),
            'stage_rho':round(float(rho.statistic),3),'stage_p':float(rho.pvalue),
            'directional':directional,'flag_underrepresented':cov<0.70,
            'apparent_in_sample':(pname=='Ours_649' and cname=='Discovery')})
        scores[(pname,cname)]=sc
bt=pd.DataFrame(rows); bt.to_csv(OUT/'benchmark_table.csv',index=False)
print(bt.to_string())

# B3 DeLong (external cohorts only, ours vs each comparator)
drows=[]
for cname in ['Fujiwara','EPoS','UCAM_Sanyal']:
    y=cohorts[cname][1]
    for pname in panels:
        if pname=='Ours_649': continue
        r=delong_test(y,scores[('Ours_649',cname)],scores[(pname,cname)])
        drows.append({'cohort':cname,'comparator':pname,'delta_AUROC':round(r[0],3),
                      'CI_lo':round(r[1],3),'CI_hi':round(r[2],3),'p':r[3]})
dl=pd.DataFrame(drows); dl.to_csv(OUT/'delong_comparisons.csv',index=False)
print(dl.to_string())

# B5 overlap
univ=set(D.index)
orows=[]
for pname,(up,dn,_) in panels.items():
    if pname=='Ours_649': continue
    pset=set(up+dn); ours=set(sig.GeneSymbol)
    inter=sorted(ours&pset); N=len(univ)
    p=st.hypergeom.sf(len(inter)-1,N,len(pset&univ),len(ours&univ))
    j=len(inter)/len(ours|pset)
    # mini-panel AUROC in externals
    minis={}
    for cname in ['Fujiwara','EPoS','UCAM_Sanyal']:
        if not inter: minis[cname]=None; continue
        z=zs(cohorts[cname][0])
        sc=z.loc[[g for g in inter if g in z.index]].mean(axis=0)
        a,lo,hi=boot_auc(cohorts[cname][1],sc)
        minis[cname]=(round(a,3),round(lo,3),round(hi,3))
    orows.append({'comparator':pname,'panel_size':len(pset),'intersection':len(inter),
        'jaccard':round(j,4),'hypergeo_p':float(p),'genes':';'.join(inter),
        'mini_AUROC_Fujiwara':minis.get('Fujiwara'),'mini_AUROC_EPoS':minis.get('EPoS'),'mini_AUROC_UCAM':minis.get('UCAM_Sanyal')})
ov=pd.DataFrame(orows); ov.to_csv(OUT/'overlap_analysis.csv',index=False)
print(ov[['comparator','panel_size','intersection','jaccard','hypergeo_p']].to_string())

# B6 parsimony
full=pd.read_csv('results/ws1_signature/ws1_dge_full.csv').set_index('GeneSymbol')
tstat=full.loc[sig.GeneSymbol,'t'].abs().sort_values(ascending=False)
prows=[]
for k in [10,25,50,100,250,649]:
    genes=tstat.head(k).index.tolist()
    up=[g for g in genes if g in set(sig[sig.logFC>0].GeneSymbol)]
    dn=[g for g in genes if g in set(sig[sig.logFC<0].GeneSymbol)]
    for cname,(df,y,stage) in cohorts.items():
        if cname=='Discovery': continue
        z=zs(df); sc=score_panel(z,up,dn,True)
        a,lo,hi=boot_auc(y,sc)
        prows.append({'size':k,'cohort':cname,'AUROC':round(a,3),'CI_lo':round(lo,3),'CI_hi':round(hi,3)})
pa=pd.DataFrame(prows); pa.to_csv(OUT/'parsimony_curve.csv',index=False)
print(pa.pivot(index='size',columns='cohort',values='AUROC').to_string())
json.dump({'protocol':'locked WS1 normalisation; seed 42; 2000 bootstrap',
  'n_panels':len(panels),'cohorts':{k:int(v[1].shape[0]) for k,v in cohorts.items()}},
  open(OUT/'stats_ws6_partial.json','w'))
