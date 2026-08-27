# B1: rebuild discovery cohort; age only from genuine GEO age fields.
# GSE135251 has NO age field in GEO (verified: characteristics are nas/fibrosis stage/
# group/disease/Stage only) — age stays NA for that cohort rather than being imputed.
suppressPackageStartupMessages({library(GEOquery); library(dplyr)})

meta <- read.csv("MASLD/harmonized_MASLD_metadata.csv", stringsAsFactors = FALSE) %>%
  filter(batch %in% c("GSE135251", "GSE130970", "GSE185051"))

fetch <- function(gse) {
  g <- getGEO(gse, GSEMatrix = TRUE, destdir = "cache")[[1]]
  pd <- pData(g)
  # genuine age/sex columns only: must mention age/sex but not 'stage'/'page' etc.
  age_col <- grep("age", colnames(pd), ignore.case = TRUE, value = TRUE)
  age_col <- age_col[!grepl("stage", age_col, ignore.case = TRUE)]
  sex_col <- grep("sex|gender", colnames(pd), ignore.case = TRUE, value = TRUE)
  data.frame(
    sample_id = rownames(pd),
    age_geo = if (length(age_col)) suppressWarnings(as.numeric(pd[[age_col[1]]])) else NA_real_,
    sex_geo = if (length(sex_col)) as.character(pd[[sex_col[1]]]) else NA_character_,
    stringsAsFactors = FALSE
  )
}
geo <- do.call(rbind, lapply(c("GSE135251", "GSE130970", "GSE185051"), fetch))

coh <- meta %>%
  left_join(geo, by = "sample_id") %>%
  mutate(age = ifelse(!is.na(age), age, age_geo),
         sex = ifelse(!is.na(sex) | is.na(sex_geo), sex, sex_geo),
         cohort = batch,
         fibrosis_group = case_when(fibrosis_stage <= 2 ~ "Early", fibrosis_stage >= 3 ~ "Late")) %>%
  select(sample_id, cohort, fibrosis_stage, age, sex, fibrosis_group)

cat("Stage alone per cohort:\n"); print(table(coh$cohort))
cat("Stage + age per cohort:\n"); print(table(coh$cohort[!is.na(coh$age)]))
cat("Early/Late, stage alone:\n"); print(table(coh$fibrosis_group))
cat("Early/Late, stage + age:\n"); print(table(coh$fibrosis_group[!is.na(coh$age)]))
write.csv(coh, "data/discovery_cohort_349.csv", row.names = FALSE)
cat("Wrote", nrow(coh), "staged rows (age NA where GEO has none)\n")
