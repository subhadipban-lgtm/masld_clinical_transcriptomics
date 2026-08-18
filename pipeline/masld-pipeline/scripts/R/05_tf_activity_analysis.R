# --- SCRIPT 3.3 (REFINED): DoRothEA TF Activity & Ferroptosis Heatmap ---

# --- 1. Robust Library Installation & Loading ---
if (!require("BiocManager", quietly = TRUE)) install.packages("BiocManager")

# List of required packages
bioc_pkgs <- c("edgeR", "dorothea", "viper", "limma")
cran_pkgs <- c("tidyverse", "pheatmap")

# Install missing Bioconductor packages
new_bioc <- bioc_pkgs[!(bioc_pkgs %in% installed.packages()[,"Package"])]
if(length(new_bioc)) BiocManager::install(new_bioc)

# Install missing CRAN packages
new_cran <- cran_pkgs[!(cran_pkgs %in% installed.packages()[,"Package"])]
if(length(new_cran)) install.packages(new_cran)

# Load libraries
library(edgeR)
library(dorothea)
library(viper)
library(tidyverse)
library(limma)
library(pheatmap)

# --- 2. Load and Prepare Data ---
message("Loading datasets...")

# 2.1 Load Expression Data
counts <- read.csv("harmonized_MASLD_RNAseq_counts.csv", row.names = 1, check.names = FALSE)

# Normalize counts using edgeR (log2 CPM)
dge <- DGEList(counts = counts)
keep <- filterByExpr(dge)
dge <- dge[keep, , keep.lib.sizes = FALSE]
dge <- calcNormFactors(dge)
expr_matrix <- cpm(dge, log = TRUE)

# 2.2 Load Metadata and GSVA Scores
metadata <- read.csv("harmonized_MASLD_metadata.csv")
gsva_scores <- read.csv("harmonized_ferroptosis_GSVA_scores.csv")
ferro_drivers <- read.csv("ferroptosis_driver.csv")
ferro_suppressors <- read.csv("ferroptosis_suppressor.csv")

# 2.3 Merge and Filter Metadata
metadata_extended <- metadata %>%
  inner_join(gsva_scores, by = c("sample_id" = "SampleID")) %>%
  filter(!is.na(fibrosis_stage)) %>%
  filter(fibrosis_stage %in% c(0, 1, 3, 4)) %>%
  mutate(fibrosis_group = case_when(
    fibrosis_stage %in% c(0, 1) ~ "Early",
    fibrosis_stage %in% c(3, 4) ~ "Late"
  ))

# Align matrix with metadata
common_samples <- intersect(colnames(expr_matrix), metadata_extended$sample_id)
expr_matrix <- expr_matrix[, common_samples]
analysis_meta <- metadata_extended %>%
  filter(sample_id %in% common_samples) %>%
  arrange(match(sample_id, common_samples))

# --- 3. Calculate TF Activity Scores ---
message("Calculating TF activity with DoRothEA...")
data(dorothea_hs, package = "dorothea")
regulons <- dorothea_hs %>%
  filter(confidence %in% c("A", "B", "C")) %>%
  df2regulon()

tf_activity <- viper(expr_matrix, regulons, eset.filter = FALSE, method = "scale", verbose = FALSE)

# --- 4. Differential TF Activity Analysis ---
message("Finding top differentially active TFs...")
design <- model.matrix(~ 0 + fibrosis_group, data = analysis_meta)
colnames(design) <- c("Early", "Late")
contrast_matrix <- makeContrasts(Late_vs_Early = Late - Early, levels = design)

fit <- lmFit(tf_activity[, analysis_meta$sample_id], design)
fit2 <- contrasts.fit(fit, contrast_matrix)
fit2 <- eBayes(fit2)

# Get top 25 TFs for a high-quality heatmap
top_results <- topTable(fit2, number = 25, coef = "Late_vs_Early")
top_tfs <- rownames(top_results)

# --- 5. Prepare Heatmap Annotations ---

# Column Annotation (Samples)
# Using dplyr::select to avoid conflicts with other packages
annotation_col <- analysis_meta %>%
  dplyr::select(sample_id, fibrosis_stage, fibrosis_group, age, GSVA_Drivers_Score, GSVA_Suppressors_Score) %>%
  column_to_rownames("sample_id")

# Row Annotation (TFs) - identify if the TFs are ferroptosis regulators
tf_info <- data.frame(TF = top_tfs) %>%
  mutate(Ferroptosis_Role = case_when(
    TF %in% ferro_drivers$symbol ~ "Driver",
    TF %in% ferro_suppressors$symbol ~ "Suppressor",
    TRUE ~ "Other"
  )) %>%
  column_to_rownames("TF")

# Define color schemes
ann_colors <- list(
  fibrosis_group = c(Early = "#7FC97F", Late = "#BEAED4"),
  fibrosis_stage = c("0"="#F7FBFF", "1"="#DEEBF7", "3"="#9ECAE1", "4"="#084594"),
  Ferroptosis_Role = c(Driver = "#E41A1C", Suppressor = "#377EB8", Other = "grey90"),
  GSVA_Drivers_Score = colorRampPalette(c("white", "red"))(100),
  GSVA_Suppressors_Score = colorRampPalette(c("white", "blue"))(100)
)

# --- 6. Generate Heatmap ---
message("Generating final heatmap...")
heatmap_data <- tf_activity[top_tfs, analysis_meta$sample_id]

# Save heatmap to file
pheatmap(
  heatmap_data,
  annotation_col = annotation_col,
  annotation_row = tf_info,
  annotation_colors = ann_colors,
  main = "TF Activity: Fibrosis Stage, Age & Ferroptosis Status",
  show_colnames = FALSE,
  scale = "row",
  cluster_cols = TRUE,
  cluster_rows = TRUE,
  fontsize_row = 9,
  color = colorRampPalette(c("navy", "white", "firebrick3"))(100),
  filename = "TF_Activity_Ferroptosis_Heatmap.png",
  width = 12,
  height = 10
)

message("✅ Analysis complete! Heatmap saved as 'TF_Activity_Ferroptosis_Heatmap.png'")
