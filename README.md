# MASLD-CDSS: Curated Database Browser and Analysis Pipeline

This repository contains (1) a curated drug–gene interaction database and browser
inputs for hypothesis generation, and (2) the complete, numbered analysis pipeline
for the accompanying manuscript.

**Manuscript:** *An early, suppressor-biased ferroptosis response precedes matrisome
convergence in MASLD fibrosis: a multi-cohort transcriptional benchmarking study*
(Banerjee, Charoensup, Vanden Berghe). DOI: [to be added on deposit].

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
                       fgsea, WS15 locked-signature build, WS27 per-stage-pair
                       limma-trend contrasts
  scripts/python/      numbered Python stages (09–53): GNN baselines, trial
                       benchmarks, deconvolution, WS26–WS31 ferroptosis
                       follow-ups (stage-resolved GSEA, paired biopsies,
                       cell-death comparison, co-expression network with
                       permutation null, single-cell effector localisation),
                       and the manuscript build script
```

Scripts are numbered by workflow stage (WS); each writes `results/stats_*.json`
provenance alongside its outputs. Random seeds are fixed and recorded.

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
DrugCentral, DGIdb, FerrDb).
