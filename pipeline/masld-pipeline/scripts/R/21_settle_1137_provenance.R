# B2: forensic re-run to determine which cohort produced DGE_All_Genes_Results.csv
suppressPackageStartupMessages({library(limma); library(edgeR); library(dplyr); library(GEOquery)})
setwd("/Users/subhadipbanerjee/masld-revision")

ent <- read.delim("data/kamzolas/gene_symbols_to_entrez.tsv", stringsAsFactors=FALSE)
ent <- ent[!is.na(ent$entrez_id) & !is.na(ent$gene_symbol) & !duplicated(ent$entrez_id),]
ent_map <- setNames(ent$gene_symbol, as.character(ent$entrez_id))

gses <- c("GSE135251","GSE130970","GSE185051","GSE167523","GSE126848")
cl <- lapply(gses, function(g){
  f <- list.files("zenodo_upload/raw_data", pattern=paste0("^",g), full.names=TRUE)
  df <- read.delim(gzfile(f), sep="\t", header=TRUE, row.names=1, check.names=FALSE,
                   stringsAsFactors=FALSE); df})
names(cl) <- gses
common <- Reduce(intersect, lapply(cl, rownames))
mat <- as.matrix(do.call(cbind, lapply(cl, function(d) as.matrix(d[common,]))))
storage.mode(mat) <- "numeric"
sym <- ent_map[rownames(mat)]; keep <- !is.na(sym)
mat <- mat[keep,]; sym <- sym[keep]
o <- order(rowMeans(mat), decreasing=TRUE); mat <- mat[o,]; sym <- sym[o]
kd <- !duplicated(sym); mat <- mat[kd,]; rownames(mat) <- sym[kd]
cat("matrix:", nrow(mat), "genes x", ncol(mat), "samples\n")

disc <- read.csv("data/discovery_cohort_349.csv", stringsAsFactors=FALSE)
cat("staged samples missing from count matrix:",
    paste(setdiff(disc$sample_id, colnames(mat)), collapse=", "), "\n")
cat("Early/Late among matched:", "\n"); print(table(disc$fibrosis_group[disc$sample_id %in% colnames(mat)]))
run_fit <- function(samples, group, ages=NULL, tag){
  m <- mat[, samples]
  log_cpm <- log2(edgeR::cpm(edgeR::DGEList(m)) + 1)
  if (is.null(ages)) { d <- model.matrix(~0+group); } else { d <- model.matrix(~0+group+ages) }
  colnames(d) <- make.names(colnames(d))
  cn <- makeContrasts(Late.Early=groupLate-groupEarly, levels=d)
  fit <- eBayes(contrasts.fit(lmFit(log_cpm, d), cn))
  res <- topTable(fit, number=Inf)
  res$GeneSymbol <- rownames(res)
  n_deg <- sum(res$adj.P.Val<0.05 & abs(res$logFC)>0.5)
  cat(sprintf("%s: n=%d genes=%d DEG(padj&lfc)=%d\n", tag, length(samples), nrow(res), n_deg))
  write.csv(res, file.path("results/rebuild_stages", paste0("dge_", tag, ".csv")), row.names=FALSE)
  res
}

# (a) clean 351, ~group
s <- intersect(disc$sample_id, colnames(mat)); d1 <- disc[match(s, disc$sample_id),]
a <- run_fit(s, factor(d1$fibrosis_group, levels=c("Early","Late")), NULL, "clean351_nocovar")
# (b) clean with age (n=135), ~group+age
d2 <- d1[!is.na(d1$age),]
b <- run_fit(d2$sample_id, factor(d2$fibrosis_group, levels=c("Early","Late")), d2$age, "clean135_age")
# (c) contaminated: + NASH samples mapped to Late (GSE167523 NASH=47, GSE126848 disease NASH=16)
pd1 <- pData(getGEO("GSE167523", GSEMatrix=TRUE, destdir="cache")[[1]])
pd2 <- pData(getGEO("GSE126848", GSEMatrix=TRUE, destdir="cache")[[1]])
n1 <- rownames(pd1)[pd1[["disease subtype:ch1"]]=="NASH"]
n2 <- rownames(pd2)[pd2[["disease:ch1"]]=="NASH"]
add <- intersect(c(n1, n2), colnames(mat)); cat("NASH-mapped additions found:", length(add), "\n")
grp <- c(d1$fibrosis_group, rep("Late", length(add))); ss <- c(d1$sample_id, add)
c1 <- run_fit(ss, factor(grp, levels=c("Early","Late")), NULL, "contaminated_nocovar")
# (d) contaminated variant with age (GSE167523 NASH have age; GSE126848 no age field)
ag <- c(d1$age, pd1[n1,"age:ch1"][n1 %in% add], rep(NA, sum(add %in% n2)))
cc <- data.frame(s=ss, g=grp, a=suppressWarnings(as.numeric(ag))) |> filter(!is.na(a))
c2 <- run_fit(cc$s, factor(cc$g, levels=c("Early","Late")), cc$a, "contaminated_age")

# ---- comparison to existing file ----
old <- read.csv("MASLD/unified_analysis_outputs/DGE_All_Genes_Results.csv", stringsAsFactors=FALSE)
old_deg <- old$GeneSymbol[old$adj.P.Val<0.05 & abs(old$logFC)>0.5]
old_deg <- old_deg[old_deg %in% rownames(mat)]
cat("\nOld file: nrow", nrow(old), "DEG(threshold, in mat):", length(old_deg), "\n")
for (nm in c("clean351_nocovar","clean135_age","contaminated_nocovar","contaminated_age")) {
  new <- read.csv(file.path("results/rebuild_stages", paste0("dge_", nm, ".csv")), stringsAsFactors=FALSE)
  m <- merge(old, new, by="GeneSymbol", suffixes=c(".old",".new"))
  sp <- cor(m$logFC.old, m$logFC.new, method="spearman")
  nd <- new$GeneSymbol[new$adj.P.Val<0.05 & abs(new$logFC)>0.5]
  ov <- length(intersect(old_deg, nd)); ji <- ov/length(union(old_deg, nd))
  cat(sprintf("%s: shared=%d spearman_logFC=%.4f oldDEG=%d newDEG=%d overlap=%d jaccard=%.4f\n",
      nm, nrow(m), sp, length(old_deg), length(nd), ov, ji))
}
