#!/usr/bin/env bash
# ============================================================================
# MASLD Drug Prediction Master Pipeline — Orchestration Script
# ============================================================================
# 
# Refactored: now reads config.yaml and exports all thresholds as
# environment variables so that R and Python scripts can use them.
#
# Usage:
#   chmod +x run_pipeline.sh
#   ./run_pipeline.sh [stages]
#   ./run_pipeline.sh 1 2 3          # run only stages 1-3
#   ./run_pipeline.sh --diag        # run only C7 diagnostics
# ============================================================================

set -euo pipefail

# ---- Parse optional stage arguments ----
RUN_STAGES="${*:-1 2 3 4 5 6 7 8 9}"
RUN_DIAG=false
if [[ "${1:-}" == "--diag" ]]; then
  RUN_DIAG=true
  shift
  RUN_STAGES=""
fi

# ---- Color codes ----
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m'

# ---- Paths ----
PIPELINE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${PIPELINE_ROOT}/data"
RESULTS_DIR="${PIPELINE_ROOT}/results"
R_SCRIPTS="${PIPELINE_ROOT}/scripts/R"
PY_SCRIPTS="${PIPELINE_ROOT}/scripts/python"
CONFIG_FILE="${PIPELINE_ROOT}/config.yaml"

mkdir -p "${DATA_DIR}/raw" "${DATA_DIR}/processed" "${RESULTS_DIR}"

# ============================================================================
# LOAD CONFIG → EXPORT AS ENV VARS
# ============================================================================
echo -e "${BLUE}Loading config from ${CONFIG_FILE}${NC}"

_parse_yaml_val() {
  # Extract a top-level scalar value from config.yaml
  # Usage: _parse_yaml_val KEY < config.yaml
  python3 -c "
import sys, yaml
with open('${CONFIG_FILE}') as f:
    cfg = yaml.safe_load(f)
val = cfg.get('${1}')
if isinstance(val, list):
    print(','.join(str(v) for v in val))
else:
    print(val if val is not None else '')
" 2>/dev/null || echo ""
}

export LOG_FC_CUTOFF=$(_parse_yaml_val LOG_FC_CUTOFF)
export P_ADJ_CUTOFF=$(_parse_yaml_val P_ADJ_CUTOFF)
export WGCNA_BETA=$(_parse_yaml_val WGCNA_BETA)
export WGCNA_MIN_MODULE_SIZE=$(_parse_yaml_val WGCNA_MIN_MODULE_SIZE)
export WGCNA_MERGE_CUT_HEIGHT=$(_parse_yaml_val WGCNA_MERGE_CUT_HEIGHT)
export GNN_DIMS=$(_parse_yaml_val GNN_DIMS)
export EPOCHS=$(_parse_yaml_val EPOCHS)
export PATIENCE=$(_parse_yaml_val PATIENCE)
export LEARNING_RATE=$(_parse_yaml_val LEARNING_RATE)
export WEIGHT_DECAY=$(_parse_yaml_val WEIGHT_DECAY)
export DROPOUT=$(_parse_yaml_val DROPOUT)
export SEEDS=$(_parse_yaml_val SEEDS)
export DEFAULT_SEED=$(_parse_yaml_val DEFAULT_SEED)
export VAL_SIZE=$(_parse_yaml_val VAL_SIZE)
export TEST_SIZE=$(_parse_yaml_val TEST_SIZE)
export BALANCE_DATASET=$(_parse_yaml_val BALANCE_DATASET)
export DEGREE_PRESERVING_NEGATIVES=$(_parse_yaml_val DEGREE_PRESERVING_NEGATIVES)
export USE_2HOP_SAMPLING=$(_parse_yaml_val USE_2HOP_SAMPLING)
export LOCO_EPOCHS=$(_parse_yaml_val LOCO_EPOCHS)
export EXTERNAL_ACCESSION=$(_parse_yaml_val EXTERNAL_ACCESSION)

# Make sure at least defaults exist
: "${LOG_FC_CUTOFF:=0.5}"
: "${P_ADJ_CUTOFF:=0.05}"
: "${EPOCHS:=200}"
: "${PATIENCE:=20}"
: "${GNN_DIMS:=128,64,32}"
: "${SEEDS:=0,1,2,3,4}"
: "${DEFAULT_SEED:=42}"

echo -e "${GREEN}  LOG_FC_CUTOFF=${LOG_FC_CUTOFF}  P_ADJ_CUTOFF=${P_ADJ_CUTOFF}${NC}"
echo -e "${GREEN}  GNN_DIMS=${GNN_DIMS}  EPOCHS=${EPOCHS}  SEEDS=${SEEDS}${NC}"
echo ""

# ============================================================================
# HEADER
# ============================================================================
echo -e "${PURPLE}============================================================================${NC}"
echo -e "${PURPLE}  MASLD Drug Prediction Master Pipeline${NC}"
echo -e "${PURPLE}  Personalised GraphSAGE GNN for Ferroptosis-Driven Fibrosis${NC}"
echo -e "${PURPLE}============================================================================${NC}"
echo ""
echo -e "${BLUE}Pipeline Root: ${PIPELINE_ROOT}${NC}"
echo -e "${BLUE}Started at: $(date)${NC}"
echo ""

# ---- Helper: run a stage if requested ----
_run_stage() {
  local stage_num="$1"
  if echo "${RUN_STAGES}" | grep -qw "${stage_num}"; then
    return 0
  else
    return 1
  fi
}

# ============================================================================
# DIAGNOSTICS-ONLY MODE
# ============================================================================
if $RUN_DIAG; then
  echo -e "${GREEN}>>> DIAGNOSTICS MODE: Running C7 embedding diagnostic only${NC}"
  conda run --no-capture-output -n masld-env python "${PY_SCRIPTS}/15_embedding_diagnostics.py" \
    --gexf "${PIPELINE_ROOT}/${DATA_DIR}/masld_personalized_kg_enhanced.gexf" \
    --weights "${RESULTS_DIR}/masld_personalized_kg_enhanced.pt" \
    --save-dir "${RESULTS_DIR}"
  echo -e "${GREEN}    ✓ Diagnostics complete${NC}"
  exit 0
fi

# ============================================================================
# STAGE 1: Metadata Harmonization (R)
# ============================================================================
if _run_stage 1; then
  echo -e "${GREEN}>>> STAGE 1: Metadata Harmonization${NC}"
  echo -e "${YELLOW}    Downloading and harmonizing metadata from 5 GEO MASLD cohorts${NC}"
  cd "${DATA_DIR}/raw"
  Rscript "${R_SCRIPTS}/01_metadata_harmonise.R"
  echo -e "${GREEN}    ✓ Metadata harmonized${NC}"
  echo ""
fi

# ============================================================================
# STAGE 2: Batch Correction & Data Integration (R)
# ============================================================================
if _run_stage 2; then
  echo -e "${GREEN}>>> STAGE 2: Batch Correction & Data Integration${NC}"
  echo -e "${YELLOW}    Combining 5 cohorts + ComBat-seq batch correction${NC}"
  Rscript "${R_SCRIPTS}/02_combine_correct_masld.R"
  echo -e "${GREEN}    ✓ Batch-corrected matrix saved${NC}"
  echo ""
fi

# ============================================================================
# STAGE 3: Network Analysis & DGE (R)
# ============================================================================
if _run_stage 3; then
  echo -e "${GREEN}>>> STAGE 3: Differential Gene Expression & Pathway Enrichment${NC}"
  echo -e "${YELLOW}    limma DGE → GO/KEGG/Reactome/MSigDB${NC}"
  Rscript "${R_SCRIPTS}/03_network_analysis.R"
  echo -e "${GREEN}    ✓ DGE results + PPI network + enrichment outputs${NC}"
  echo ""
fi

# ============================================================================
# STAGE 4: Ferroptosis Analysis (R)
# ============================================================================
if _run_stage 4; then
  echo -e "${GREEN}>>> STAGE 4: Ferroptosis Gene Analysis${NC}"
  echo -e "${YELLOW}    GSVA scores + GSEA + ferroptosis DEG visualization${NC}"
  Rscript "${R_SCRIPTS}/04_ferroptosis_analysis.R"
  echo -e "${GREEN}    ✓ Ferroptosis analysis complete${NC}"
  echo ""
fi

# ============================================================================
# STAGE 5: TF Activity Analysis (R)
# ============================================================================
if _run_stage 5; then
  echo -e "${GREEN}>>> STAGE 5: Transcription Factor Activity (DoRothEA/VIPER)${NC}"
  Rscript "${R_SCRIPTS}/05_tf_activity_analysis.R"
  echo -e "${GREEN}    ✓ TF activity heatmap saved${NC}"
  echo ""
fi

# ============================================================================
# STAGE 6: Score Correlation Analysis (R)
# ============================================================================
if _run_stage 6; then
  echo -e "${GREEN}>>> STAGE 6: Correlation Analysis${NC}"
  Rscript "${R_SCRIPTS}/06_score_correlation.R"
  echo -e "${GREEN}    ✓ Correlation results saved${NC}"
  echo ""
fi

# ============================================================================
# STAGE 7: Heatmap Visualization (R)
# ============================================================================
if _run_stage 7; then
  echo -e "${GREEN}>>> STAGE 7: Publication-Grade Heatmaps${NC}"
  Rscript "${R_SCRIPTS}/07_masld_heatmaps.R"
  echo -e "${GREEN}    ✓ Heatmaps generated${NC}"
  echo ""
fi

# ============================================================================
# STAGE 8: WGCNA Visualization (R)
# ============================================================================
if _run_stage 8; then
  echo -e "${GREEN}>>> STAGE 8: WGCNA Module Visualization${NC}"
  Rscript "${R_SCRIPTS}/08_wgcna_visualization.R"
  echo -e "${GREEN}    ✓ WGCNA Figure 8 panels generated${NC}"
  echo ""
fi

# ============================================================================
# STAGE 9: GraphSAGE GNN Pipeline (Python)
# ============================================================================
if _run_stage 9; then
  echo -e "${GREEN}>>> STAGE 9: GraphSAGE GNN Drug Prediction${NC}"
  echo -e "${YELLOW}    Knowledge Graph construction + GNN training + inductive evaluation${NC}"
  echo -e "${YELLOW}    Using config: GNN_DIMS=${GNN_DIMS}  EPOCHS=${EPOCHS}  SEEDS=${SEEDS}${NC}"
  cd "${PIPELINE_ROOT}"

  # Run the refactored pipeline stages
  # 9a: Build KG (if enhance_knowledge_graph.py exists)
  if [[ -f "${PY_SCRIPTS}/09a_build_kg.py" ]]; then
    conda run --no-capture-output -n masld-env python "${PY_SCRIPTS}/09a_build_kg.py"
  fi

  # 9b: Train GNN (main training script)
  if [[ -f "${PY_SCRIPTS}/09b_train_gnn.py" ]]; then
    conda run --no-capture-output -n masld-env python "${PY_SCRIPTS}/09b_train_gnn.py" \
      --config "${CONFIG_FILE}" \
      --save-dir "${RESULTS_DIR}"
  fi

  # 9c: Evaluate (LOCO + baselines)
  if [[ -f "${PY_SCRIPTS}/09c_evaluate.py" ]]; then
    conda run --no-capture-output -n masld-env python "${PY_SCRIPTS}/09c_evaluate.py" \
      --config "${CONFIG_FILE}" \
      --save-dir "${RESULTS_DIR}"
  fi

  # 9d: Explain (GNNExplainer)
  if [[ -f "${PY_SCRIPTS}/09d_explain.py" ]]; then
    conda run --no-capture-output -n masld-env python "${PY_SCRIPTS}/09d_explain.py" \
      --config "${CONFIG_FILE}" \
      --save-dir "${RESULTS_DIR}"
  fi

  # 9e: Screen drugs
  if [[ -f "${PY_SCRIPTS}/09e_screen_drugs.py" ]]; then
    conda run --no-capture-output -n masld-env python "${PY_SCRIPTS}/09e_screen_drugs.py" \
      --config "${CONFIG_FILE}" \
      --save-dir "${RESULTS_DIR}"
  fi

  echo -e "${GREEN}    ✓ GNN pipeline complete — results in ${RESULTS_DIR}/${NC}"
  echo ""
fi

# ============================================================================
# COMPLETE
# ============================================================================
echo -e "${PURPLE}============================================================================${NC}"
echo -e "${GREEN}  PIPELINE COMPLETE${NC}"
echo -e "${PURPLE}============================================================================${NC}"
echo ""
echo -e "${BLUE}Results directory: ${RESULTS_DIR}${NC}"
echo -e "${BLUE}Finished at: $(date)${NC}"
echo ""
echo -e "${YELLOW}Post-pipeline validation steps:${NC}"
echo -e "  1. Run embedding diagnostic:  ./run_pipeline.sh --diag"
echo -e "  2. Run trial benchmark:     conda run -n masld-env python ${PY_SCRIPTS}/11_trial_benchmark.py --gexf <gexf> --weights <pt>"
echo -e "  3. Run external validation:  Rscript ${R_SCRIPTS}/10_external_validation.R"
echo ""
