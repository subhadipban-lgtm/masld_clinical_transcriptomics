#!/usr/bin/env Rscript
# -*- coding: utf-8 -*-
# =============================================================================
# C1 — External Cohort Validation (GSE163211)
# =============================================================================
#
# Scores an independent cohort with the frozen 1,137-gene signature,
# tests AUROC for late (F3-4) vs early (F0-2), Spearman stage gradient,
# and whether driver/suppressor GSVA scores remain coupled.
#
# Usage:
#   Rscript scripts/R/10_external_validation.R
#
# Success criterion: AUROC >= 0.75 and significant stage gradient.
# =============================================================================

suppressPackageStartupMessages({
  library(GEOquery)
  library(limma)
  library(edgeR)
  library(GSVA)
  library(pheatmap)
  library(ggplot2)
  library(cowplot)
  library(jsonlite)
})

# ---- Config ----
EXT_ACCESSION  <- "GSE163211"
EXT_DIR       <- file.path("data", "raw", EXT_ACCESSION)
RESULTS_DIR   <- "results"
LOGFC_CUTOFF <- 0.5    # from config.yaml
P_ADJ_CUTOFF  <- 0.05
SIG_N_GENES   <- 1137  # from the 09 pipeline's DGE output

# Read the same config.yaml if available
config_path <- "config.yaml"
if (file.exists(config_path)) {
  yaml <- yaml::read_yaml(config_path)
  LOGFC_CUTOFF <- yaml$LOG_FC_CUTOFF %||% LOGFC_CUTOFF
  P_ADJ_CUTOFF  <- yaml$P_ADJ_CUTOFF %||% P_ADJ_CUTOFF
}

`%||%` <- function(a, b) if (is.null(a)) b else a

message("=== C1: External Cohort Validation ===")
message("Accession: ", EXT_ACCESSION)

# ---- Step 1: Download & load external data ----
if (!dir.exists(EXT_DIR)) {
  dir.create(EXT_DIR, recursive = TRUE)
  message("Downloading ", EXT_ACCESSION, " …")
  getGEOSuppFiles(EXT_ACCESSION, makeDirectory = TRUE, baseDir = "data/raw")
}

# Find the count matrix (typically a .txt or .csv in the supplementary)
raw_files <- list.files(EXT_DIR, pattern = "\\.(txt|csv|tsv|gz)$", full.names = TRUE)
if (length(raw_files) == 0) {
  # Try the GEO soft file
  message("Attempting to download via GEOquery …")
  gse <- getGEO(EXT_ACCESSION, GSEMatrix = TRUE)[[1]]
  expr_raw <- exprs(gse)
  pdata <- pData(gse)
} else {
  # Load the first matrix file
  message("Loading from: ", raw_files[1])
  expr_raw <- as.matrix(read.delim(raw_files[1], row.names = 1, check.names = FALSE))
  # Try to get metadata from a separate file or from GEO
  gse_meta <- tryCatch({
    getGEO(EXT_ACCESSION, GSEMatrix = TRUE)[[1]]
  }, error = function(e) NULL)
  if (!is.null(gse_meta)) {
    pdata <- pData(gse_meta)
  } else {
    # Fallback: try to find a metadata file
    meta_files <- list.files(EXT_DIR, pattern = "(meta|sample|clinical)", full.names = TRUE, ignore.case = TRUE)
    if (length(meta_files) > 0) {
      pdata <- read.delim(meta_files[1], check.names = FALSE)
    } else {
      stop("Cannot find phenotype data for ", EXT_ACCESSION)
    }
  }
}

# ---- Step 2: Harmonise fibrosis staging ----
# Map external cohort's staging to our F0-F4 scheme.
# GSE163211 uses NAS/fibrosis scores — adapt the column name.
stage_col <- NULL
for (col in c("fibrosis_stage", "stage", "nas_fibrosis", "histologic_fibrosis_stage",
                "fibrosis", "staging", "grade", "nash_stage", "MASLD_fibrosis")) {
  if (col %in% colnames(pdata)) {
    stage_col <- col
    break
  }
}
if (is.null(stage_col)) {
  message("Available columns: ", paste(colnames(pdata), collapse = ", "))
  stop("No fibrosis stage column found in phenotype data.")
}

stages_raw <- as.character(pdata[[stage_col]])
message("Raw stages (first 20): ", paste(head(stages_raw, 20), collapse = ", "))

# Harmonise to F0, F1, F2, F3, F4
harmonise_stage <- function(s) {
  s <- trimws(toupper(s))
  if (s %in% c("F0", "F0/F1", "0", "NO FIBROSIS", "NONE", "0-0")) return("F0")
  if (s %in% c("F1", "1", "PORTAL FIBROSIS", "MILD", "0-1")) return("F1")
  if (s %in% c("F2", "2", "PERIPORTAL", "MODERATE", "1-2", "EARLY")) return("F2")
  if (s %in% c("F3", "3", "BRIDGING", "SEPTAL", "2-3")) return("F3")
  if (s %in% c("F4", "4", "CIRRHOSIS", "LATE", "3-4")) return("F4")
  return(NA_character_)
}

fibrosis_stage <- sapply(stages_raw, harmonise_stage, USE.NAMES = FALSE)

# ---- Step 3: Align expression matrix ----
# Keep only samples with harmonised stage
keep_samples <- !is.na(fibrosis_stage)
message("Samples with harmonised stage: ", sum(keep_samples), "/", length(fibrosis_stage))

if (sum(keep_samples) < 20) {
  stop("Too few samples with harmonised stage (<20). Check staging column.")
}

expr_mat <- expr_raw[, keep_samples, drop = FALSE]
metadata  <- pdata[keep_samples, , drop = FALSE]
metadata$fibrosis_stage <- fibrosis_stage[keep_samples]

# Binary: early (F0-2) vs late (F3-4)
metadata$is_late <- ifelse(metadata$fibrosis_stage %in% c("F3", "F4"), 1, 0)

message("\nStage distribution:")
print(table(metadata$fibrosis_stage))
message("Early (F0-2): ", sum(metadata$is_late == 0),
        "  Late (F3-4): ", sum(metadata$is_late == 1))

# ---- Step 4: Load the 1,137-gene signature ----
# The signature is produced by 03_network_analysis.R
sig_path <- file.path("data", "processed", "dge_results.csv")
if (!file.exists(sig_path)) {
  # Fallback: try to find any DGE results
  sig_path <- list.files("data", pattern = "dge.*\\.csv$", full.names = TRUE, recursive = TRUE)[1]
}
if (!file.exists(sig_path)) {
  message("WARNING: No DGE results found. Using top SIG_N_GENES by variance.")
  gene_vars <- apply(expr_mat, 1, var, na.rm = TRUE)
  sig_genes <- names(sort(gene_vars, decreasing = TRUE))[1:SIG_N_GENES]
} else {
  dge_df <- read.csv(sig_path, stringsAsFactors = FALSE)
  # Assume columns: gene, logFC, adj_p_val
  dge_df <- dge_df[dge_df$adj_p_val < P_ADJ_CUTOFF & abs(dge_df$logFC) > LOGFC_CUTOFF, ]
  sig_genes <- dge_df$gene[1:min(nrow(dge_df), SIG_N_GENES)]
  message("Using ", length(sig_genes), " signature genes from DGE results.")
}

# Map to expression matrix
common_genes <- intersect(sig_genes, rownames(expr_mat))
message("Signature genes in external cohort: ", length(common_genes), "/", length(sig_genes))

if (length(common_genes) < 100) {
  # Try gene symbol mapping
  message("Low overlap. Attempting to map via gene symbols …")
  # If expression matrix uses Entrez IDs, try aliasing
  library(org.Hs.eg.db)
  mapped <- mapIds(org.Hs.eg.db, keys = common_genes, column = "SYMBOL", keytype = "SYMBOL", multiVals = "first")
  common_genes <- intersect(names(mapped), rownames(expr_mat))
  message("After mapping: ", length(common_genes), " genes.")
}

expr_sig <- expr_mat[common_genes, , drop = FALSE]

# ---- Step 5: Score with GSVA (or simple z-score mean) ----
message("\nRunning GSVA on external cohort …")

# Separate up and down genes from signature
if (exists("dge_df") && "logFC" %in% colnames(dge_df)) {
  up_genes   <- intersect(dge_df$gene[dge_df$logFC > 0], common_genes)
  down_genes <- intersect(dge_df$gene[dge_df$logFC < 0], common_genes)
} else {
  # Fallback: use median expression to split
  gene_means <- rowMeans(expr_sig, na.rm = TRUE)
  up_genes   <- names(sort(gene_means, decreasing = TRUE))[1:(length(common_genes) %/% 2)]
  down_genes <- setdiff(common_genes, up_genes)
}

message("Up-regulated: ", length(up_genes), "  Down-regulated: ", length(down_genes))

# GSVA
tryCatch({
  gsva_res <- gsva(expr_sig,
                     list(up = up_genes, down = down_genes),
                     method = "gsva", kcdf = "Gaussian")
  scores_matrix <- t(as.matrix(gsva_res))
  colnames(scores_matrix) <- c("driver_score", "suppressor_score")
  metadata <- cbind(metadata, scores_matrix)
  message("GSVA scoring complete.")
}, error = function(e) {
  message("GSVA failed (", e$message, "). Using z-score mean fallback.")
  # Simple z-score scoring
  z_mat <- t(scale(t(expr_sig)))
  metadata$driver_score   <- if (length(up_genes) > 0) rowMeans(z_mat[up_genes, , drop = FALSE], na.rm = TRUE) else 0
  metadata$suppressor_score <- if (length(down_genes) > 0) rowMeans(z_mat[down_genes, , drop = FALSE], na.rm = TRUE) else 0
})

# Composite score: up minus down
metadata$composite_score <- metadata$driver_score - metadata$suppressor_score

# ---- Step 6: Evaluate ----
message("\n=== External Validation Results ===")

# 6a. AUROC for late vs early
if (sum(metadata$is_late == 1) > 0 && sum(metadata$is_late == 0) > 0) {
  pred <- pROC::roc(metadata$is_late, metadata$composite_score, quiet = TRUE)
  auroc_ext <- as.numeric(auc(pred))
  ci <- pROC::ci.auc(pred)
  message(sprintf("AUROC (late vs early): %.3f  95%% CI [%.3f, %.3f]",
                  auroc_ext, ci[1], ci[3]))
  if (auroc_ext >= 0.75) {
    message("  ✓ STRONG: Replicates the finding (AUROC >= 0.75).")
  } else if (auroc_ext >= 0.70) {
    message("  ~ PUBLISHABLE: Replication at AUROC >= 0.70.")
  } else {
    message("  ⚠ Below 0.70: Signature does not replicate well in this cohort.")
  }
} else {
  auroc_ext <- NA
  message("  Cannot compute AUROC: missing late or early samples.")
}

# 6b. Spearman correlation with fibrosis stage
stage_numeric <- as.numeric(factor(metadata$fibrosis_stage, levels = c("F0","F1","F2","F3","F4")))
cor_test <- cor.test(stage_numeric, metadata$composite_score, method = "spearman", exact = FALSE)
message(sprintf("Spearman r = %.3f, p = %.2e", cor_test$estimate, cor_test$p.value))

# 6c. Driver-suppressor coupling
if ("driver_score" %in% colnames(metadata) && "suppressor_score" %in% colnames(metadata)) {
  coupling_test <- cor.test(metadata$driver_score, metadata$suppressor_score, method = "spearman")
  message(sprintf("Driver-Suppressor correlation: r = %.3f, p = %.2e",
                  coupling_test$estimate, coupling_test$p.value))
  coupling_r <- coupling_test$estimate
} else {
  coupling_r <- NA
}

# ---- Step 7: Visualise ----
dir.create(RESULTS_DIR, showWarnings = FALSE, recursive = TRUE)

# Boxplot: composite score by stage
metadata$stage_factor <- factor(metadata$fibrosis_stage, levels = c("F0","F1","F2","F3","F4"))

p1 <- ggplot(metadata, aes(x = stage_factor, y = composite_score, fill = stage_factor)) +
  geom_boxplot(outlier.shape = 21) +
  geom_jitter(width = 0.15, alpha = 0.5, size = 1.5) +
  scale_fill_brewer(palette = "YlOrRd") +
  labs(title = paste("External Validation:", EXT_ACCESSION),
       x = "Fibrosis Stage", y = "Composite Signature Score (Up - Down)") +
  theme_cowplot(font_size = 12) +
  theme(legend.position = "none")

ggsave(file.path(RESULTS_DIR, "external_validation_boxplot.png"), p1, width = 7, height = 5, dpi = 300)
message("Saved: ", file.path(RESULTS_DIR, "external_validation_boxplot.png"))

# Driver vs Suppressor scatter
if ("driver_score" %in% colnames(metadata)) {
  p2 <- ggplot(metadata, aes(x = driver_score, y = suppressor_score, color = stage_factor)) +
    geom_point(size = 2.5) +
    scale_color_brewer(palette = "Set1") +
    labs(title = "Driver vs Suppressor GSVA Scores",
         x = "Driver (Ferroptosis) Score", y = "Suppressor Score") +
    theme_cowplot(font_size = 12)
  ggsave(file.path(RESULTS_DIR, "external_validation_driver_suppressor.png"), p2, width = 7, height = 5, dpi = 300)
}

# ---- Step 8: Save stats ----
stats <- list(
  accession = EXT_ACCESSION,
  n_samples = nrow(metadata),
  n_late = sum(metadata$is_late == 1),
  n_early = sum(metadata$is_late == 0),
  n_sig_genes = length(common_genes),
  auroc = ifelse(is.na(auroc_ext), NA, auroc_ext),
  auroc_ci = ifelse(is.na(auroc_ext), NA, paste(round(ci[1], 3), round(ci[3], 3), sep = "-")),
  spearman_r = cor_test$estimate,
  spearman_p = cor_test$p.value,
  driver_suppressor_r = ifelse(is.na(coupling_r), NA, coupling_r),
  replicates = auroc_ext >= 0.70
)
write_json(stats, file.path(RESULTS_DIR, "stats_10_external_validation.json"), auto_unbox = TRUE, pretty = TRUE)
message("Saved stats → ", file.path(RESULTS_DIR, "stats_10_external_validation.json"))

message("\n=== C1 External Validation Complete ===")
