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
colnames(counts_all) <- unlist(lapply(count_list, colnames))

# Map Entrez IDs to HGNC symbols
symbols <- ent_map[rownames(counts_all)]
valid <- !is.na(symbols)
counts_all <- counts_all[valid, ]
symbols <- symbols[valid]

# Deduplicate
row_means <- rowMeans(counts_all)
ord <- order(row_means, decreasing=TRUE)
counts_all <- counts_all[ord, ]
symbols <- symbols[ord]
keep_sym <- !duplicated(symbols)
counts_all <- counts_all[keep_sym, ]
rownames(counts_all) <- symbols[keep_sym]

# 3. Load metadata from GEO
gse135 <- getGEO('GSE135251', GSEMatrix = TRUE)[[1]]
pd135 <- pData(gse135)
meta135 <- data.frame(
  sample_id = rownames(pd135),
  stage = as.numeric(pd135[['fibrosis stage:ch1']]),
  batch = 'GSE135251',
  stringsAsFactors = FALSE
)

gse130 <- getGEO('GSE130970', GSEMatrix = TRUE)[[1]]
pd130 <- pData(gse130)
meta130 <- data.frame(
  sample_id = rownames(pd130),
  stage = as.numeric(pd130[['fibrosis stage:ch1']]),
  batch = 'GSE130970',
  stringsAsFactors = FALSE
)

gse185 <- getGEO('GSE185051', GSEMatrix = TRUE)[[1]]
pd185 <- pData(gse185)
meta185 <- data.frame(
  sample_id = rownames(pd185),
  stage = as.numeric(pd185[['fibrosis_stage:ch1']]),
  batch = 'GSE185051',
  stringsAsFactors = FALSE
)

meta_df <- rbind(meta135, meta130, meta185)
meta_df <- meta_df[!is.na(meta_df$stage), ]
meta_df$group <- ifelse(meta_df$stage <= 2, 'Early', 'Late')

common_samples <- intersect(colnames(counts_all), meta_df$sample_id)
counts_mat <- counts_all[, common_samples]
meta_df <- meta_df[match(common_samples, meta_df$sample_id), ]

cat(paste('Total samples in contrast:', ncol(counts_mat), '\n'))
cat('Sample breakdown by batch and stage:\n')
print(table(meta_df$batch, meta_df$stage))
cat('Sample breakdown by Early (F0-F2) vs Late (F3-F4):\n')
print(table(meta_df$batch, meta_df$group))

# Expressed filter: CPM > 1 in >= 10% of samples
cpm_mat <- t(t(counts_mat) / colSums(counts_mat) * 1e6)
keep_genes <- rowSums(cpm_mat > 1) >= (0.1 * ncol(counts_mat))
log_cpm <- log2(cpm_mat[keep_genes, ] + 1)
cat(paste('Expressed genes tested on full genome:', nrow(log_cpm), '\n'))

# Limma DGE
design <- model.matrix(~ 0 + group + batch, data = meta_df)
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
cat(paste('Genes tested on full genome:', nrow(res), '\n'))
cat(paste('Unadjusted P < 0.05:', raw_p_05, '\n'))
cat(paste('FDR padj < 0.05 alone:', padj_05, '\n'))
cat(paste('|log2FC| > 0.5 alone:', fc_05, '\n'))
cat(paste('INTERSECTION (padj < 0.05 & |log2FC| > 0.5):', both_05, '\n'))

cat('\nTop 20 DEGs by padj:\n')
print(head(res[, c('GeneSymbol', 'logFC', 'P.Value', 'adj.P.Val')], 20))
