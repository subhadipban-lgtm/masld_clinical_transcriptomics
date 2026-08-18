# MASLD DrugScope: Personalized Drug Prediction for MASLD via a Ferroptosis-Aware GraphSAGE GNN

> **Decoding the functional signature of ferroptosis-driven fibrosis with a personalised GNN**
> 
> Banerjee S¹·², Charoensupa R¹, Vanden Berghe W²
> 
> ¹ Medicinal Plant Innovation Center, Mae Fah Luang University, Chiang Rai, Thailand  
> ² Laboratory for Protein Chemistry, Proteomics & Epigenetic Signaling (PPES), University of Antwerp, Belgium

## Overview

This repository contains the complete master pipeline and web-based tool for **personalized drug prediction for Metabolic dysfunction-Associated Steatotic Liver Disease (MASLD)** — formerly NAFLD/NASH. The framework integrates:

1. **A systems-level transcriptomic meta-analysis** of 504 MASLD patients across 5 GEO cohorts
2. **A personalized GraphSAGE Graph Neural Network (GNN)** that learns functional drug-target relationships from a 3,687-node knowledge graph
3. **Stage- and age-aware personalization** based on fibrosis severity (F0-F4) and patient age

The model identifies drug candidates by learning a **functional** (not chemical) representation of drug similarity, successfully clustering known MASLD therapeutics from different pharmacological classes and identifying **Disulfiram** as a high-priority repurposing candidate.

## Key Results

| Metric | Value |
|--------|-------|
| Cohort size | 504 patients (5 GEO cohorts) |
| DEG fibrosis signature | 1,137 genes |
| Knowledge Graph | 3,687 nodes, 3,780 edges |
| GNN architecture | 2-layer GraphSAGE (15→64→32 dims) |
| Inductive test AUC (Personalised KG) | **1.000** |
| Inductive test AUC (General KG) | 0.938 |
| LOCO mean AUC (6 drug classes) | 0.966 |
| GNN vs WGCNA concordance (ARI) | 0.9459 |
| GNN vs XGBoost baseline | 1.000 vs 0.293 |
| Functional vs chemical similarity (r) | 0.30 (p=0.0002) |

## Repository Structure

```
masld-pipeline/
├── run_pipeline.sh                    # Master orchestration script
├── README.md                          # This file
├── LICENSE                            # MIT License
├── CITATION.cff                       # Citation information
├── envs/
│   ├── requirements_R.txt             # R package dependencies
│   └── requirements_python.txt        # Python package dependencies
├── scripts/
│   ├── R/
│   │   ├── 01_metadata_harmonise.R            # GEO metadata download & harmonization
│   │   ├── 02_combine_correct_masld.R         # ComBat-seq batch correction
│   │   ├── 03_network_analysis.R              # DGE + pathway enrichment + PPI
│   │   ├── 04_ferroptosis_analysis.R          # Ferroptosis GSVA + GSEA
│   │   ├── 05_tf_activity_analysis.R          # DoRothEA/VIPER TF activity
│   │   ├── 06_score_correlation.R             # Metadata × score correlation
│   │   ├── 07_masld_heatmaps.R                # Publication heatmaps
│   │   └── 08_wgcna_visualization.R           # WGCNA module visualization
│   └── python/
│       └── 09_graphsage_pipeline.py           # GraphSAGE GNN pipeline
├── data/
│   ├── raw/                           # Raw GEO data (downloaded)
│   └── processed/                     # Harmonized/processed data
├── results/                           # Pipeline outputs
└── docs/
    ├── REPRODUCIBILITY.md             # Step-by-step reproducibility guide
    └── DATA_DESCRIPTION.md            # Input/output file descriptions
```

## Pipeline Stages

### Stage 1-2: Data Harmonization (R)
Downloads and harmonizes metadata from 5 GEO MASLD cohorts. Applies ComBat-seq batch correction preserving biological variation. Output: 504 samples × 20,768 genes.

**Cohorts:** GSE126848, GSE135251, GSE167523, GSE130970, GSE185051

### Stage 3: Differential Expression & Enrichment (R)
limma DGE with `~0 + fibrosis_group + age` design. Late vs Early (F3-F4 vs F0-F2) contrast identifies 1,137-gene fibrosis signature. Multi-database ORA (GO-BP, KEGG, Reactome, MSigDB Hallmark) + GSEA for ferroptosis gene set.

### Stage 4-5: Ferroptosis & TF Activity (R)
GSVA scores for FerrDb V2 driver/suppressor gene sets. DoRothEA/VIPER TF activity identifies SMAD3, TP53, FOS/JUN, NFKB1/RELA as differentially active. Correlation analysis reveals driver-suppressor coupling (TIMP1↔TXN, R=0.80).

### Stage 6-8: WGCNA & Visualization (R)
Signed WGCNA (β=12) identifies blue module (3,201 genes, r≈0.55 with fibrosis). Publication-grade heatmaps, correlation plots, and module eigengene visualizations.

### Stage 9: GraphSAGE GNN (Python)
2-layer GraphSAGE link predictor on personalized knowledge graph. 15-dim node features (10 categorical + 5 numeric including DGE logFC). Inductive evaluation on 11 holdout drugs. LOCO CV across 6 pharmacological classes. GNNExplainer for interpretability.

## Quick Start

### Prerequisites

- **R** >= 4.2 with Bioconductor >= 3.16
- **Python** >= 3.9 with PyTorch >= 2.0 + PyTorch Geometric >= 2.3
- **RAM**: 32GB recommended (WGCNA + GNN training)
- **Disk**: ~50GB for raw GEO data
- **GPU**: Optional but recommended for GNN training (CUDA 11.8+)

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/masld-drugscope.git
cd masld-drugscope

# Install R dependencies
Rscript -e 'install.packages(c("BiocManager", "tidyverse", "dplyr", "readr", "tibble"))
BiocManager::install(c("GEOquery", "sva", "limma", "edgeR", "dorothea", "viper", "clusterProfiler", "org.Hs.eg.db", "ReactomePA", "enrichplot", "STRINGdb", "GOplot", "impute", "preprocessCore"))
install.packages(c("pheatmap", "RColorBrewer", "ggrepel", "ggpubr", "corrplot", "msigdbr", "igraph", "ggraph", "WGCNA", "patchwork"))'

# Install Python dependencies
pip install torch torch-geometric pandas numpy networkx scikit-learn matplotlib seaborn xgboost causallearn python-docx
```

### Running the Pipeline

```bash
# Run the complete pipeline
chmod +x run_pipeline.sh
./run_pipeline.sh
```

Or run individual stages:

```bash
# R stages
cd data/raw && Rscript ../../scripts/R/01_metadata_harmonise.R
Rscript scripts/R/02_combine_correct_masld.R
# ... etc

# Python GNN stage
python scripts/python/09_graphsage_pipeline.py
```

## Web Tool

The web-based tool for personalized drug prediction is available as a Next.js application. It provides an interactive interface where users input:

- **MASLD fibrosis stage** (F0-F4)
- **Patient age** (18-85 years)

And receive:

- Ranked drug recommendations with personalized match scores
- Stage-specific hypothesis (Early Intervention / Advanced Fibrosis / Pan-Stage)
- Predicted drug-target interactions with DGE features
- Age-adjusted confidence scoring
- Drug-target network visualization

### Running the Web Tool

```bash
cd web-tool
npm install
npm run dev
# Open http://localhost:3000
```

## Personalization Logic

### Fibrosis Stage Matching
The GNN predicts top-20 target genes for each drug. The average logFC (Late vs Early) of these targets determines a **stage hypothesis**:
- `avg logFC < -0.1` → **Early Intervention** (targets early-stage genes)
- `avg logFC > 0.1` → **Advanced Fibrosis** (targets late-stage genes)
- Otherwise → **Pan-Stage**

### Age Adjustment
Based on the observed **inverse correlation between age and ferroptosis scores** (the "aging paradox"):
- **Young (<40)**: Ferroptosis-modulating drugs receive a boost
- **Middle (40-60)**: No adjustment
- **Older (>60)**: Metabolic-axis drugs favored over ferroptosis modulators

## Drug Candidates

The pipeline evaluates 14 drug candidates across 8 pharmacological classes:

| Drug | Class | Status | Stage Hypothesis |
|------|-------|--------|-----------------|
| Resmetirom | THR-β Agonist | FDA-Approved (2024) | Pan-Stage |
| Liraglutide | GLP-1 RA | FDA-Approved | Early Intervention |
| Semaglutide | GLP-1 RA | FDA-Approved | Early Intervention |
| Obeticholic Acid | FXR Agonist | FDA-Approved | Early Intervention |
| **Disulfiram** | ALDH Inhibitor | **Repurposing Candidate** | **Advanced Fibrosis** |
| Pioglitazone | TZD | FDA-Approved | Early Intervention |
| Lanifibranor | Pan-PPAR | Investigational | Early Intervention |
| Resveratrol | Polyphenol | Natural Product | Pan-Stage |
| Curcumin | Polyphenol | Natural Product | Advanced Fibrosis |
| Silymarin | Polyphenol | Natural Product | Advanced Fibrosis |
| Berberine | Polyphenol | Natural Product | Pan-Stage |
| Quercetin | Flavonoid | Natural Product | Advanced Fibrosis |
| Myricetin | Flavonoid | Natural Product | Advanced Fibrosis |
| Ellagic Acid | Polyphenol | Natural Product | Advanced Fibrosis |

## Citation

If you use this pipeline or web tool in your research, please cite:

```bibtex
@article{banerjee2026masld,
  title={Decoding the functional signature of ferroptosis-driven fibrosis with a personalised GNN},
  author={Banerjee, Subhadip and Charoensupa, Rawiwan and Vanden Berghe, Wim},
  journal={[TBD]},
  year={2026},
  publisher={[TBD]}
}
```

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

## Data Availability

- **GEO datasets**: GSE126848, GSE135251, GSE167523, GSE130970, GSE185051 (publicly available from NCBI GEO)
- **FerrDb V2**: http://www.zhounan.org/ferrdb/
- **Drug-target interactions**: Curated from literature (see manuscript Supplementary Materials)

## Acknowledgments

- Mae Fah Luang University, Thailand
- University of Antwerp, Belgium
- National Center for Biotechnology Information (NCBI) GEO
- FerrDb V2 database curators

## Contact

For questions about the pipeline or collaboration inquiries:
- **Subhadip Banerjee**: [email]
- **Rawiwan Charoensupa**: [email]
- **Wim Vanden Berghe**: [email]

## Disclaimer

This tool is for **research use only** and is not intended for clinical decision-making. All drug predictions are hypothesis-generating and must be validated through appropriate clinical trials. Always consult a qualified healthcare provider before making treatment decisions.
