# ------------------------------------------------------------------
# R Script to Download and Harmonize Metadata for MASLD RNA-seq Studies
# ------------------------------------------------------------------
# This script will:
# 1. Download metadata for five specified GEO datasets.
# 2. Print the column names for each study for easier debugging.
# 3. Standardize key clinical variables (disease status, fibrosis, etc.).
# 4. Combine all standardized metadata into a single data frame.
# 5. Save the final, clean table to a CSV file.
# ------------------------------------------------------------------

# Step 1: Install and Load Required Libraries
if (!require("BiocManager", quietly = TRUE)) install.packages("BiocManager")
if (!require("GEOquery", quietly = TRUE)) BiocManager::install("GEOquery")
if (!require("dplyr", quietly = TRUE)) install.packages("dplyr")

library(GEOquery)
library(dplyr)

# Step 2: Define the list of GEO datasets to process
gse_ids <- c("GSE126848", "GSE135251", "GSE167523", "GSE130970", "GSE185051")

# Create an empty list to store the standardized metadata from each study
standardized_meta_list <- list()

# Step 3: Loop through each GEO ID, download, and standardize its metadata
for (gse_id in gse_ids) {

  cat(paste("\nProcessing", gse_id, "...\n"))

  # Download the GEO series object
  gse <- getGEO(gse_id, GSEMatrix = TRUE)
  # Extract the metadata into a data frame
  raw_metadata <- pData(gse[[1]])

  # *** DIAGNOSTIC STEP ADDED HERE ***
  # Print the column names to help with debugging any future errors.
  cat("Columns found:\n")
  print(colnames(raw_metadata))

  # Add a unique sample ID column from the row names
  raw_metadata$sample_id <- rownames(raw_metadata)

  # --- Standardization Block ---
  # This is where we apply custom logic for each specific dataset
  # because they all name their columns differently.

  if (gse_id == "GSE126848") {
    clean_meta <- raw_metadata %>%
      mutate(
        disease_status = case_when(
          grepl("healthy", source_name_ch1, ignore.case = TRUE) ~ "Healthy",
          grepl("NAFL", source_name_ch1, ignore.case = TRUE) ~ "Steatosis",
          grepl("NASH", source_name_ch1, ignore.case = TRUE) ~ "NASH"
        ),
        batch = gse_id
      ) %>%
      select(sample_id, disease_status, batch)

  } else if (gse_id == "GSE135251") {
    clean_meta <- raw_metadata %>%
      mutate(
        disease_status = `disease:ch1`,
        fibrosis_stage = as.numeric(gsub("F", "", `fibrosis stage:ch1`)),
        # Extract values by removing the text part (e.g., "age (y): ")
        age = as.numeric(gsub("age \\(y\\): ", "", characteristics_ch1)),
        sex = gsub("gender: ", "", characteristics_ch1.1),
        bmi = as.numeric(gsub("bmi: ", "", characteristics_ch1.2)),
        batch = gse_id
      ) %>%
      select(sample_id, disease_status, fibrosis_stage, age, sex, bmi, batch)

  } else if (gse_id == "GSE167523") {
    clean_meta <- raw_metadata %>%
      mutate(
        disease_status = `disease subtype:ch1`,
        fibrosis_stage = as.numeric(gsub("F", "", gsub("fibrosis stage: ", "", characteristics_ch1.1))),
        age = as.numeric(`age:ch1`),
        sex = `gender:ch1`, # Note: column is 'gender:ch1', not 'sex:ch1'
        bmi = as.numeric(gsub("bmi: ", "", characteristics_ch1)),
        batch = gse_id
      ) %>%
      select(sample_id, disease_status, fibrosis_stage, age, sex, bmi, batch)

  } else if (gse_id == "GSE130970") {
    clean_meta <- raw_metadata %>%
      mutate(
        nas_score = as.numeric(`nafld activity score:ch1`),
        disease_status = case_when(
          nas_score >= 5 ~ "NASH",
          nas_score >= 1 ~ "Steatosis",
          nas_score == 0 ~ "Healthy",
          TRUE ~ "Unknown"
        ),
        fibrosis_stage = as.numeric(gsub("Stage ", "", `fibrosis stage:ch1`)),
        age = as.numeric(`age at biopsy:ch1`),
        sex = `Sex:ch1`,
        batch = gse_id
      ) %>%
      select(sample_id, disease_status, fibrosis_stage, age, sex, batch)

  } else if (gse_id == "GSE185051") {
    # *** FIX APPLIED HERE ***
    # The error was caused by incorrect column names for disease, age, and sex,
    # and a missing column for BMI. This block now uses the correct names
    # identified from the diagnostic output.
    clean_meta <- raw_metadata %>%
      mutate(
        disease_status = `disease:ch1`,
        fibrosis_stage = as.numeric(gsub("F", "", `fibrosis_stage:ch1`)),
        age = as.numeric(`age:ch1`),
        sex = `gender:ch1`,
        batch = gse_id
      ) %>%
      select(sample_id, disease_status, fibrosis_stage, age, sex, batch)
  }

  # Add the cleaned metadata to our list
  standardized_meta_list[[gse_id]] <- clean_meta
}

# Step 4: Combine all the standardized data frames into a single master table
# bind_rows is smart and will fill missing columns with NA
harmonized_metadata <- bind_rows(standardized_meta_list)

# Step 5: Final Cleaning and Saving
# Let's make the disease_status column more consistent
harmonized_metadata <- harmonized_metadata %>%
  mutate(disease_status = case_when(
    grepl("normal|healthy", disease_status, ignore.case = TRUE) ~ "Healthy",
    grepl("nash|steatohepatitis", disease_status, ignore.case = TRUE) ~ "NASH",
    grepl("steatosis|nafld|nafl", disease_status, ignore.case = TRUE) ~ "Steatosis",
    TRUE ~ "Other" # Handle any edge cases
  ))

# Display the dimensions and a summary of the final harmonized table
cat("\n-------------------------------------------------\n")
cat("Metadata harmonization complete.\n")
cat(paste("Total samples:", nrow(harmonized_metadata), "\n"))
cat("Summary of disease status:\n")
print(table(harmonized_metadata$disease_status, useNA = "ifany"))
cat("\nSummary of fibrosis stage:\n")
print(table(harmonized_metadata$fibrosis_stage, useNA = "ifany"))
cat("\nSummary of batches (source studies):\n")
print(table(harmonized_metadata$batch, useNA = "ifany"))
cat("-------------------------------------------------\n")


# Save the final harmonized metadata to a CSV file
output_filename <- "harmonized_MASLD_metadata.csv"
write.csv(harmonized_metadata, output_filename, row.names = FALSE)

cat(paste("Final harmonized metadata saved to:", output_filename, "\n"))

