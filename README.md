# MASLD-Clinical_Transcriptomics

This repository contains the complete, numbered analysis pipeline
for the accompanying manuscript.

**Manuscript:** *Independent hepatic transcriptional signatures converge on the
matrisome and fail at the F2 treatment boundary in MASLD fibrosis: a multi-cohort
benchmarking study* (Banerjee, Charoensup, Vanden Berghe). DOI: [to be added on
deposit].

**Headline findings:** independently derived MASLD fibrosis signatures — five
published panels and our own — converge on a shared matrisome programme that is
present from the earliest stage transition, rises with fibrosis, and replicates in
an independent cohort; every signature fails at the F2 treatment-eligibility
boundary, a biological ceiling of bulk transcriptomics. The ferroptosis programme,
evaluated under a size-matched random-set null guard, does not carry staging
information beyond genome-wide expression drift and is reported as a
well-characterised negative.

## Repository layout

```
data/                  curated databases and reference sets (FerrDb V2 filtered
                       driver/suppressors with provenance, expression matrix,
                       scored metadata)
pipeline/masld-pipeline/
  scripts/R/           numbered R stages (01-15):
                       01  5-cohort harmonisation, count merge, limma DGE
                       02  3-cohort discovery DGE replication
                       03  full-transcriptome DGE (~group+batch)
                       04  full-genome DGE (~group+age+batch)
                       05  build 349-sample discovery cohort from GEO
                       06  forensic DGE provenance (1137-gene origin)
                       07  alternative DGE fits, ComBat-seq test
                       08  locked 649-gene signature via voom+limma
                       09  WGCNA co-expression modules
                       10  fgsea on ranked logFC, random null sets
                       11  signature enrichment, paediatric sensitivity
                       12  fgseaMultilevel on pre-ranked t-stat
                       13  hypergeometric matrisome/ECM enrichment
                       14  per-stage-pair DGE, discovery
                       15  per-stage-pair DGE, Fujiwara
  scripts/python/      numbered Python stages (01-17) + utilities:
                       01  external validation (Fujiwara, EPoS, UCAM)
                       02  decisive permutation test (10k perms)
                       03  paired-biopsy longitudinal (58 pairs)
                       04  single-cell pseudobulk (GSE136103)
                       05  parse drug targets (CTD/DrugBank/DrugCentral)
                       06  panel benchmark (vs published signatures)
                       07  bulk deconvolution (NNLS)
                       08  ferroptosis scoring, bootstrap AUROC
                       09  per-stage-pair GSEA, paired trajectory
                       10  co-expression network, permutation null
                       11  cell-death pathway stage association
                       12  single-cell effector localisation
                       13  matrisome GSEA replication
                       14  Fujiwara GSEA, NABA matrisome
                       15  Fujiwara direct mean-z score test
                       16  random-set null + covariate adjustment
                       17  null-guarded pathway table (7 pathways)
                       build_manuscript.py / build_manuscript_v2.py
                           manuscript DOCX builders
```

Scripts are numbered in execution order; each writes `results/stats_*.json`
provenance alongside its outputs. Random seeds are fixed and recorded, and every
mean-z pathway score is reported against a 1,000-set size-matched random null.

## Installation and usage

```bash
# R (>= 4.2): limma, edgeR, fgsea, WGCNA
# Python (>= 3.9): pandas, numpy, scipy, statsmodels, scikit-learn,
#                  matplotlib, seaborn, networkx, python-docx
```

Run stages in numeric order; each script documents its required inputs in the
module docstring and fails loudly when an input is missing — there are no
silent fallbacks.

## Citation

Please cite the accompanying manuscript (DOI above) and this repository.

## License

MIT (code); database contents retain their upstream licences (CTD, DrugBank,
DrugCentral, DGIdb, FerrDb, MSigDB/QuickGO, GEO).
