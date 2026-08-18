# ------------------------------------------------------------------
# R Script to Combine, Batch Correct, and Save Harmonized RNA-seq Data
# ------------------------------------------------------------------
# This script will:
# 1. Load the harmonized clinical metadata file created previously.
# 2. Load the five raw gene count files.
# 3. Find common genes across all datasets.
# 4. Combine the datasets into a single count matrix.
# 5. Align the count matrix with the metadata.
# 6. Apply ComBat-seq to correct for batch effects.
# 7. Save the final, analysis-ready harmonized count matrix.
# ------------------------------------------------------------------

# Step 1: Install and Load Required Libraries
if (!require("BiocManager", quietly = TRUE)) install.packages("BiocManager")
if (!require("sva", quietly = TRUE)) BiocManager::install("sva")
if (!require("dplyr", quietly = TRUE)) install.packages("dplyr")
if (!require("readr", quietly = TRUE)) install.packages("readr")
if (!require("tibble", quietly = TRUE)) install.packages("tibble")

library(sva)
library(dplyr)
library(readr)
library(tibble)


# Step 2: Load the Harmonized Metadata
# This file must be in the same directory as this script.
metadata_file <- "harmonized_MASLD_metadata.csv"
if (!file.exists(metadata_file)) {
  stop("Error: The harmonized metadata file was not found. Please run the metadata script first.")
}
harmonized_metadata <- read_csv(metadata_file)

# Step 3: Load the Raw Count Files
# List the filenames for the raw count data.
count_files <- c(
  "GSE126848_raw_counts_GRCh38.p13_NCBI.tsv",
  "GSE135251_raw_counts_GRCh38.p13_NCBI.tsv",
  "GSE167523_raw_counts_GRCh38.p13_NCBI.tsv",
  "GSE130970_raw_counts_GRCh38.p13_NCBI.tsv",
  "GSE185051_raw_counts_GRCh38.p13_NCBI.tsv"
)

# Read all files into a list of data frames
# We set the first column ('GeneID') as the row names.
count_data_list <- lapply(count_files, function(file) {
  cat(paste("Reading", file, "...\n"))
  read_tsv(file) %>%
    as.data.frame() %>%
    column_to_rownames("GeneID")
})

cat("\nAll count files loaded successfully.\n")

# Step 4: Combine the Expression Data
# Find the common genes that are present in all five datasets
common_genes <- Reduce(intersect, lapply(count_data_list, rownames))
cat(paste("Found", length(common_genes), "common genes across all datasets.\n"))

# Filter each dataset to only include common genes and then combine them
combined_counts <- do.call(cbind, lapply(count_data_list, function(df) {
  df[common_genes, ]
}))

cat(paste("Combined count matrix created with", nrow(combined_counts), "genes and", ncol(combined_counts), "samples.\n"))

# Step 5: Align the Metadata and Count Matrix
# This is a critical step. The order of columns in the count matrix
# must exactly match the order of rows in the metadata.
harmonized_metadata <- harmonized_metadata %>%
  filter(sample_id %in% colnames(combined_counts))

combined_counts <- combined_counts[, harmonized_metadata$sample_id]

# Ensure the alignment is perfect
stopifnot(all(colnames(combined_counts) == harmonized_metadata$sample_id))
cat("Metadata and count matrix are now perfectly aligned.\n")

# Step 6: Apply ComBat-seq for Batch Correction
# ComBat-seq cannot handle NAs in the model variables.
# We will filter out any samples with 'Other' or NA in disease_status.
valid_samples <- !is.na(harmonized_metadata$disease_status) & harmonized_metadata$disease_status != "Other"
filtered_metadata <- harmonized_metadata[valid_samples, ]
filtered_counts <- combined_counts[, valid_samples]

# Define the batch (the study each sample came from)
batch <- filtered_metadata$batch

# *** FIX APPLIED HERE ***
# The 'modcombat' argument is deprecated in newer versions of the sva package.
# Instead, the biological variable to preserve is passed directly to the 'group' argument.
# The model.matrix line is no longer needed.

cat("\nRunning ComBat-seq to correct for batch effects. This may take a few minutes...\n")

# Run the batch correction algorithm.
corrected_counts <- ComBat_seq(
  counts = as.matrix(filtered_counts),
  batch = batch,
  group = filtered_metadata$disease_status # Use 'group' to preserve biological variation
)

cat("Batch correction complete!\n")

# Step 7: Save the Final Harmonized Files
output_filename <- "harmonized_MASLD_RNAseq_counts.csv"
# We add the GeneID back as the first column for the final file
final_output <- as.data.frame(corrected_counts) %>%
  rownames_to_column("GeneID")

write.csv(final_output, output_filename, row.names = FALSE)

cat("\n-------------------------------------------------\n")
cat("Harmonization process finished.\n")
cat(paste("Final batch-corrected expression data saved to:", output_filename, "\n"))
cat("You can now use this file and the metadata file for your downstream Python analysis.\n")
cat("-------------------------------------------------\n")

