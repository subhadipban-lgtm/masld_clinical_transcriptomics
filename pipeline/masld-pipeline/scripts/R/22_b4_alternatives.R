# B4: alternative fits vs the existing DGE file; ComBat inflation test.
suppressPackageStartupMessages({library(limma); library(edgeR); library(dplyr); library(sva)})
setwd("/Users/subhadipbanerjee/masld-revision")
ent <- read.delim("data/kamzolas/gene_symbols_to_entrez.tsv", stringsAsFactors=FALSE)
ent <- ent[!is.na(ent$entrez_id) & !is.na(ent$gene_symbol) & !duplicated(ent$entrez_id),]
gses <- c("GSE135251","GSE130970","GSE185051")
cl <- lapply(gses, function(g){ f <- list.files("zenodo_upload/raw_data", pattern=paste0("^",g), full.names=TRUE)
  read.delim(gzfile(f), sep="\t", header=TRUE, row.names=1, check.names=FALSE, stringsAsFactors=FALSE)})
common <- Reduce(intersect, lapply(cl, rownames))
mat <- as.matrix(do.call(cbind, lapply(cl, function(d) as.matrix(d[common,])))); storage.mode(mat) <- "numeric"
sym <- ent$gene_symbol[match(rownames(mat), as.character(ent$entrez_id))]
keep <- !is.na(sym); mat <- mat[keep,]; sym <- sym[keep]
o <- order(rowMeans(mat), decreasing=TRUE); mat <- mat[o,]; sym <- sym[o]
kd <- !duplicated(sym); mat <- mat[kd,]; rownames(mat) <- sym[kd]

disc <- read.csv("data/discovery_cohort_349.csv", stringsAsFactors=FALSE)
disc <- disc[match(intersect(disc$sample_id, colnames(mat)), disc$sample_id),]
m <- mat[, disc$sample_id]
grp <- factor(disc$fibrosis_group, levels=c("Early","Late"))
coh <- factor(disc$cohort)

dge <- DGEList(m); dge <- calcNormFactors(dge)
log_cpm <- log2(cpm(dge) + 1)

old <- read.csv("MASLD/unified_analysis_outputs/DGE_All_Genes_Results.csv", stringsAsFactors=FALSE)
old_deg_all <- old$GeneSymbol[old$adj.P.Val<0.05 & abs(old$logFC)>0.5]
report <- function(res, tag){
  mm <- merge(old, data.frame(GeneSymbol=rownames(res), logFC=res$logFC), by="GeneSymbol")
  sp <- cor(mm$logFC.x, mm$logFC.y, method="spearman")
  nd <- sum(res$adj.P.Val<0.05 & abs(res$logFC)>0.5)
  ov <- length(intersect(old_deg_all, rownames(res)[res$adj.P.Val<0.05 & abs(res$logFC)>0.5]))
  cat(sprintf("%s: spearman=%.4f DEG=%d overlap_with_1137=%d\n", tag, sp, nd, ov))
  res
}
fit_of <- function(expr, design){
  cn <- makeContrasts(Late.Early=groupLate-groupEarly, levels=design)
  topTable(eBayes(contrasts.fit(lmFit(expr, design), cn)), number=Inf)
}
# A: ~ group, no ComBat
dA <- model.matrix(~0+grp); colnames(dA) <- c("groupEarly","groupLate")
rA <- report(fit_of(log_cpm, dA), "A_group_349_logCPM")
# B: ~ group + cohort
dB <- model.matrix(~0+grp+coh); colnames(dB)[1:2] <- c("groupEarly","groupLate")
rB <- report(fit_of(log_cpm, dB), "B_group_cohort_349")
# C: ComBat-seq (batch=cohort, as in script 02) then logCPM, ~ group
comb <- ComBat_seq(as.matrix(m), batch=as.integer(coh))
dgec <- calcNormFactors(DGEList(comb)); lcpm_c <- log2(cpm(dgec) + 1)
rC <- report(fit_of(lcpm_c, dA), "C_group_349_combat")
# extra: ComBat with age-like eBayes settings already default; also voom variant for reference
rD <- report(fit_of(voom(dge, dA)$E, dA), "D_group_349_voomE")
cat("\nDEG counts at padj<0.05 & |log2FC|>0.5 — inflation test:\n")
cat("logCPM, no ComBat (A):", sum(rA$adj.P.Val<0.05 & abs(rA$logFC)>0.5), "\n")
cat("ComBat-seq then logCPM (C):", sum(rC$adj.P.Val<0.05 & abs(rC$logFC)>0.5), "\n")
