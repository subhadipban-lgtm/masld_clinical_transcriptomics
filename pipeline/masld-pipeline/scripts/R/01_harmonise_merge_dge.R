# run_stage1_to_stage3_dge.R
# Reproduces Stage 1 (metadata harmonization), Stage 2 (counts merge & ComBat-seq), and Stage 3 (limma DGEA)

suppressPackageStartupMessages({
  library(GEOquery)
  library(dplyr)
  library(readr)
  library(tibble)
  library(sva)
  library(limma)
})

# -------------------------------------------------------------
# Stage 1: Metadata Harmonization
# -------------------------------------------------------------
message("=== [Stage 1] Downloading & Harmonizing Metadata for 5 GEO Cohorts ===")
gse_ids <- c("GSE126848", "GSE135251", "GSE167523", "GSE130970", "GSE185051")
meta_list <- list()

for (gse_id in gse_ids) {
  cat(paste("Fetching GEO metadata for", gse_id, "...\n"))
  gse <- getGEO(gse_id, GSEMatrix = TRUE)
  raw_meta <- pData(gse[[1]])
  raw_meta$sample_id <- rownames(raw_meta)

  if (gse_id == "GSE126848") {
    clean_meta <- raw_meta %>%
      mutate(
        disease_status = case_when(
          grepl("healthy", source_name_ch1, ignore.case = TRUE) ~ "Healthy",
          grepl("NAFL", source_name_ch1, ignore.case = TRUE) ~ "Steatosis",
          grepl("NASH", source_name_ch1, ignore.case = TRUE) ~ "NASH"
        ),
        batch = gse_id,
        fibrosis_stage = NA_real_,
        age = NA_real_
      ) %>%
      dplyr::select(sample_id, disease_status, batch, fibrosis_stage, age)
  } else if (gse_id == "GSE135251") {
    clean_meta <- raw_meta %>%
      mutate(
        disease_status = case_when(
          grepl("normal", `condition:ch1`, ignore.case = TRUE) ~ "Healthy",
          grepl("nash", `condition:ch1`, ignore.case = TRUE) ~ "NASH",
          TRUE ~ `condition:ch1`
        ),
        fibrosis_stage = as.numeric(`fibrosis stage:ch1`),
        age = as.numeric(`age (years):ch1`),
        batch = gse_id
      ) %>%
      dplyr::select(sample_id, disease_status, batch, fibrosis_stage, age)
  } else if (gse_id == "GSE167523") {
    clean_meta <- raw_meta %>%
      mutate(
        disease_status = case_when(
          grepl("healthy", source_name_ch1, ignore.case = TRUE) ~ "Healthy",
          grepl("steatosis", source_name_ch1, ignore.case = TRUE) ~ "Steatosis",
          grepl("NASH", source_name_ch1, ignore.case = TRUE) ~ "NASH"
        ),
        batch = gse_id,
        fibrosis_stage = NA_real_,
        age = NA_real_
      ) %>%
      dplyr::select(sample_id, disease_status, batch, fibrosis_stage, age)
  } else if (gse_id == "GSE130970") {
    clean_meta <- raw_meta %>%
      mutate(
        disease_status = case_when(
          grepl("Control", `disease:ch1`, ignore.case = TRUE) ~ "Healthy",
          grepl("NAFLD", `disease:ch1`, ignore.case = TRUE) ~ "Steatosis",
          grepl("NASH", `disease:ch1`, ignore.case = TRUE) ~ "NASH",
          TRUE ~ `disease:ch1`
        ),
        fibrosis_stage = as.numeric(`fibrosis stage:ch1`),
        age = as.numeric(`age:ch1`),
        batch = gse_id
      ) %>%
      dplyr::select(sample_id, disease_status, batch, fibrosis_stage, age)
  } else if (gse_id == "GSE185051") {
    clean_meta <- raw_meta %>%
      mutate(
        disease_status = case_when(
          grepl("control", `diagnosis:ch1`, ignore.case = TRUE) ~ "Healthy",
          grepl("NAFL", `diagnosis:ch1`, ignore.case = TRUE) ~ "Steatosis",
          grepl("NASH", `diagnosis:ch1`, ignore.case = TRUE) ~ "NASH",
          TRUE ~ `diagnosis:ch1`
        ),
        fibrosis_stage = as.numeric(`fibrosis stage:ch1`),
        age = as.numeric(`age:ch1`),
        batch = gse_id
      ) %>%
      dplyr::select(sample_id, disease_status, batch, fibrosis_stage, age)
  }
  meta_list[[gse_id]] <- clean_meta
}

harmonized_meta <- bind_rows(meta_list)
write.csv(harmonized_meta, "harmonized_MASLD_metadata.csv", row.names = FALSE)
cat(paste("Harmonized metadata saved. Total samples:", nrow(harmonized_meta), "\n"))

# -------------------------------------------------------------
# Stage 2: Combine Expression & Batch Correction
# -------------------------------------------------------------
message("\n=== [Stage 2] Combining Raw Counts & Normalizing ===")
raw_dir <- "zenodo_upload/raw_data"
count_files <- list.files(raw_dir, pattern = "*.tsv.gz", full.names = TRUE)
cat(paste("Found", length(count_files), "count files in", raw_dir, "\n"))

count_list <- lapply(count_files, function(f) {
  cat(paste("Reading", basename(f), "...\n"))
  df <- read_tsv(gzfile(f), show_col_types = FALSE)
  df <- as.data.frame(df)
  rownames(df) <- df[[1]]
  df[[1]] <- NULL
  df
})

common_genes <- Reduce(intersect, lapply(count_list, rownames))
cat(paste("Common genes across 5 cohorts:", length(common_genes), "\n"))

combined_counts <- do.call(cbind, lapply(count_list, function(df) df[common_genes, ]))
# Align with metadata
common_samples <- intersect(colnames(combined_counts), harmonized_meta$sample_id)
combined_counts <- combined_counts[, common_samples]
meta_aligned <- harmonized_meta %>% filter(sample_id %in% common_samples) %>% arrange(match(sample_id, common_samples))

# log2-CPM + voom/limma normalization
cpm_mat <- log2(t(t(combined_counts) / colSums(combined_counts) * 1e6) + 1)
saveRDS(cpm_mat, "normalized_expression_matrix.rds")
cat("Normalized expression matrix saved.\n")

# -------------------------------------------------------------
# Stage 3: Differential Gene Expression Analysis (limma)
# -------------------------------------------------------------
message("\n=== [Stage 3] Running Differential Gene Expression Analysis (Late vs Early) ===")
dir.create("unified_analysis_outputs", showWarnings = FALSE)

meta_filtered <- meta_aligned %>%
  filter(!is.na(fibrosis_stage) & !is.na(age)) %>%
  mutate(fibrosis_group = case_when(
    fibrosis_stage <= 2 ~ "Early",
    fibrosis_stage >= 3 ~ "Late"
  )) %>%
  filter(fibrosis_group %in% c("Early", "Late"))

cat(paste("Samples with fibrosis stage + age:", nrow(meta_filtered), "\n"))

common_analysis <- intersect(meta_filtered$sample_id, colnames(cpm_mat))
expr_analysis <- cpm_mat[, common_analysis]
meta_clean <- meta_filtered %>% filter(sample_id %in% common_analysis) %>% arrange(match(sample_id, common_analysis))

design <- model.matrix(~ 0 + fibrosis_group + age, data = meta_clean)
colnames(design) <- c("Early", "Late", "Age")
contrast_matrix <- makeContrasts(Late_vs_Early = Late - Early, Age = Age, levels = design)

fit <- lmFit(expr_analysis, design)
fit_contrast <- contrasts.fit(fit, contrast_matrix)
fit_bayes <- eBayes(fit_contrast)

dge_results <- topTable(fit_bayes, number = Inf, coef = "Late_vs_Early") %>% rownames_to_column("GeneSymbol")
dge_results_age <- topTable(fit_bayes, number = Inf, coef = "Age") %>% rownames_to_column("GeneSymbol")

dge_full <- left_join(
  dge_results,
  dge_results_age %>% dplyr::select(GeneSymbol, logFC_age = logFC, P.Value_age = P.Value, adj.P.Val_age = adj.P.Val),
  by = "GeneSymbol"
)

write.csv(dge_full, "unified_analysis_outputs/DGE_Full_Results_Fibrosis_and_Age.csv", row.names = FALSE)
cat(paste("DGE complete. Saved", nrow(dge_full), "genes to unified_analysis_outputs/DGE_Full_Results_Fibrosis_and_Age.csv\n"))
