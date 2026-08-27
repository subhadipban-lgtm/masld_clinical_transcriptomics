"""Task 1: parse drug-target sources, merge, dedupe. Sources: CTD, DrugBank (DB.xlsx),
DrugCentral (human only). Excluded: BindingDB 'Natural .xlsx' (target names are protein
names, no HGNC gene symbols - cannot map without fabrication); DrugComb csv (drug-drug
synergy scores, no gene targets); string_human_ppi gz (duplicate of STRING v12 already in KG)."""
import pandas as pd, re
from pathlib import Path
SRC=Path('data/Drug_protein')
frames=[]; report=[]

ctd=pd.read_excel(SRC/'CTD.xlsx',sheet_name=0)
ctd=ctd[ctd['Organism']=='Homo sapiens']
t=pd.DataFrame({'drug_name':ctd['ChemicalName'].astype(str).str.strip(),
                'target_gene':ctd['GeneSymbol'].astype(str).str.strip().str.upper(),
                'relation':ctd['InteractionActions'].astype(str).str.split('^|,').str[0].str.strip().str.lower(),
                'source':'CTD'})
t['relation']=t['relation'].replace({'decreases^expression':'represses_expression','increases^expression':'activates_expression'}).fillna('targets')
frames.append(t); report.append(('CTD.xlsx',len(ctd),{'drug_name':'ChemicalName','target_gene':'GeneSymbol','relation':'InteractionActions'}))

db=pd.read_excel(SRC/'DB.xlsx',sheet_name=0)
t=pd.DataFrame({'drug_name':db['name'].astype(str).str.strip(),
                'target_gene':db['HGNC'].astype(str).str.strip().str.upper().str.replace('HGNC:','',regex=False),
                'relation':'targets','source':'DrugBank'})
frames.append(t); report.append(('DB.xlsx',len(db),{'drug_name':'name','target_gene':'HGNC','relation':'(none -> targets)'}))

dc=pd.read_excel(SRC/'DrugCentral.xlsx',sheet_name=0)
dc=dc[dc['ORGANISM']=='Homo sapiens']
t=pd.DataFrame({'drug_name':dc['DRUG_NAME'].astype(str).str.strip(),
                'target_gene':dc['GENE'].astype(str).str.strip().str.upper(),
                'relation':dc['ACTION_TYPE'].fillna('targets').astype(str).str.strip().str.lower(),
                'source':'DrugCentral'})
frames.append(t); report.append(('DrugCentral.xlsx',len(dc),{'drug_name':'DRUG_NAME','target_gene':'GENE','relation':'ACTION_TYPE'}))

for r in report: print(r[0],'rows',r[1],'mapping',r[2])
m=pd.concat(frames,ignore_index=True)
m=m[(m.drug_name!='nan')&(m.target_gene!='nan')&(m.drug_name!='')&(m.target_gene!='')]
m=m.drop_duplicates(subset=['drug_name','target_gene'])
m.to_csv('data/kg/drug_target_merged.csv',index=False)
print('MERGED unique (drug,gene) pairs:',len(m),'| unique drugs:',m.drug_name.nunique(),'| unique genes:',m.target_gene.nunique())
m.to_csv('results/kg_additions_drug_target.csv',index=False)
