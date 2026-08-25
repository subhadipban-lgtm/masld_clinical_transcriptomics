# MASLD-CDSS: Curated Database Browser and Analysis Pipeline

This repository contains (1) a curated drug–gene interaction database and browser
inputs for hypothesis generation, and (2) the complete, numbered analysis pipeline
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

**Disclaimer:** For research and hypothesis generation only. No therapeutic
recommendations are made. The browser performs lookup of curated interactions
(CTD, DrugBank, DrugCentral, DGIdb) for user-supplied gene lists; it applies no
predictive model.

## Repository layout

```
data/                  curated databases and reference sets (FerrDb V2 filtered
                       driver/suppressors with provenance, expression matrix,
                       scored metadata)
pipeline/masld-pipeline/
  scripts/R/           numbered R stages (01–50): harmonisation, DGE, WGCNA,
                       fgsea, WS15 locked-signature build, per-stage-pair
                       limma-trend contrasts (WS27, discovery; WS30, Fujiwara)
  scripts/python/      numbered Python stages (09–53): GNN baselines, trial
                       benchmarks, deconvolution, WS26–WS31 follow-ups
                       (ferroptosis scoring with random-set null guard,
                       stage-resolved GSEA in both cohorts, paired-biopsy
                       trajectory, drift-adjusted cell-death pathway table,
                       permutation-nulled co-expression network, single-cell
                       effector localisation across donors), and the
                       revised-manuscript build script
```

Scripts are numbered by workflow stage (WS); each writes `results/stats_*.json`
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
