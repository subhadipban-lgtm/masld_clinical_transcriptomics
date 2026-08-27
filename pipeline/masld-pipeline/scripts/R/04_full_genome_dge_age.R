suppressPackageStartupMessages({
  library(GEOquery)
  library(dplyr)
  library(tibble)
  library(readr)
  library(limma)
})

# 1. Load Entrez to Symbol map
ent_tab <- read.delim('data/kamzolas/gene_symbols_to_entrez.tsv', sep='\t', header=TRUE, stringsAsFactors=FALSE)
ent_tab <- ent_tab[!is.na(ent_tab$entrez_id) & !is.na(ent_tab$gene_symbol), ]
ent_tab <- ent_tab[!duplicated(ent_tab$entrez_id), ]
ent_map <- setNames(as.character(ent_tab$gene_symbol), as.character(ent_tab$entrez_id))

# 2. Load count tables for the 3 contributing cohorts
raw_dir <- 'zenodo_upload/raw_data'
gses <- c('GSE135251', 'GSE130970', 'GSE185051')
count_list <- list()
for (gse in gses) {
  f <- list.files(raw_dir, pattern = paste0('^', gse), full.names = TRUE)
  df <- as.data.frame(read.delim(gzfile(f), sep='\t', header=TRUE, row.names=1, stringsAsFactors=FALSE, check.names=FALSE))
  count_list[[gse]] <- df
}

common_entrez <- Reduce(intersect, lapply(count_list, rownames))
counts_all <- do.call(cbind, lapply(count_list, function(df) df[common_entrez, ]))

# Map to gene symbols
symbols <- ent_map[rownames(counts_all)]
valid <- !is.na(symbols)
counts_all <- counts_all[valid, ]
symbols <- symbols[valid]

# Deduplicate symbols by keeping highest mean count
row_means <- rowMeans(counts_all)
ord <- order(row_means, decreasing=TRUE)
counts_all <- counts_all[ord, ]
symbols <- symbols[ord]
keep_sym <- !duplicated(symbols)
counts_all <- counts_all[keep_sym, ]
rownames(counts_all) <- symbols[keep_sym]

# 3. Load metadata from GEO for the 3 cohorts
meta_list <- list()
for (gse_id in gses) {
  gse <- getGEO(gse_id, GSEMatrix = TRUE)
  raw_meta <- pData(gse[[1]])
  raw_meta$sample_id <- rownames(raw_meta)
  
  # Find age column
  age_col <- grep("age", colnames(raw_meta), value=TRUE, ignore.case=TRUE)[1]
  stage_col <- grep("fibrosis stage|Stage", colnames(raw_meta), value=TRUE, ignore.case=TRUE)[1]
  
  clean <- raw_meta %>%
    mutate(
      stage = as.numeric(gsub(".*: ", "", .data[[stage_col]])),
      age = as.numeric(gsub(".*: ", "", .data[[age_col]])),
      batch = gse_id
    ) %>%
    dplyr::select(sample_id, stage, age, batch)
  
  meta_list[[gse_id]] <- clean
}
meta_df <- bind_rows(meta_list) %>%
  filter(!is.na(stage) & !is.na(age)) %>%
  mutate(group = ifelse(stage <= 2, 'Early', 'Late'))

common_samples <- intersect(colnames(counts_all), meta_df$sample_id)
counts_mat <- counts_all[, common_samples]
meta_df <- meta_df %>% filter(sample_id %in% common_samples) %>% arrange(match(sample_id, common_samples))

cat(paste('Total samples in contrast:', ncol(counts_mat), 'Early:', sum(meta_df$group == 'Early'), 'Late:', sum(meta_df$group == 'Late'), '\n'))

# Filter expressed genes (CPM > 1 in >= 10% samples)
cpm_mat <- t(t(counts_mat) / colSums(counts_mat) * 1e6)
keep_genes <- rowSums(cpm_mat > 1) >= (0.1 * ncol(counts_mat))
log_cpm <- log2(cpm_mat[keep_genes, ] + 1)
cat(paste('Expressed genes tested:', nrow(log_cpm), '\n'))

# Limma DGE
design <- model.matrix(~ 0 + group + age + batch, data = meta_df)
colnames(design) <- make.names(colnames(design))
contrast <- makeContrasts(Late_vs_Early = groupLate - groupEarly, levels = design)

fit <- lmFit(log_cpm, design)
fit_c <- contrasts.fit(fit, contrast)
fit_b <- eBayes(fit_c)

res <- topTable(fit_b, number = Inf) %>% rownames_to_column('GeneSymbol')

padj_05 <- sum(res$adj.P.Val < 0.05)
fc_05 <- sum(abs(res$logFC) > 0.5)
both_05 <- sum(res$adj.P.Val < 0.05 & abs(res$logFC) > 0.5)
raw_p_05 <- sum(res$P.Value < 0.05)

cat('\n=== FULL GENOME DGE RESULTS ON CONTRIBUTING COHORTS ===\n')
cat(paste('Genes tested:', nrow(res), '\n'))
cat(paste('Unadjusted P < 0.05:', raw_p_05, '\n'))
cat(paste('FDR padj < 0.05 alone:', padj_05, '\n'))
cat(paste('|log2FC| > 0.5 alone:', fc_05, '\n'))
cat(paste('INTERSECTION (padj < 0.05 & |log2FC| > 0.5):', both_05, '\n'))

cat('\nTop 15 DEGs by padj:\n')
print(head(res[, c('GeneSymbol', 'logFC', 'P.Value', 'adj.P.Val')], 15))
