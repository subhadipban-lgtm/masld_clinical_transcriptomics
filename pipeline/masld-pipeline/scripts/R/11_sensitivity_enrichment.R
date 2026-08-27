# Manuscript inputs: (a) enrichment of the 649-gene signature; (b) sensitivity:
# DGE excluding the paediatric cohort; (c) re-state voom vs plain-limma counts.
suppressPackageStartupMessages({library(limma); library(edgeR); library(fgsea); library(msigdbr)})
setwd("/Users/subhadipbanerjee/masld-revision"); set.seed(42)
disc <- read.csv("data/discovery_cohort_349.csv", stringsAsFactors=FALSE)
ent <- read.delim("data/kamzolas/gene_symbols_to_entrez.tsv", stringsAsFactors=FALSE)
ent <- ent[!is.na(ent$entrez_id) & !is.na(ent$gene_symbol) & !duplicated(ent$entrez_id),]
gses <- c("GSE135251","GSE130970","GSE185051")
cl <- lapply(gses, function(g){ f <- list.files("zenodo_upload/raw_data", pattern=paste0("^",g), full.names=TRUE)
  read.delim(gzfile(f), sep="\t", header=TRUE, row.names=1, check.names=FALSE, stringsAsFactors=FALSE)})
common <- Reduce(intersect, lapply(cl, rownames))
M <- as.matrix(do.call(cbind, lapply(cl, function(d) as.matrix(d[common,])))); storage.mode(M) <- "numeric"
rownames(M) <- ent$gene_symbol[match(rownames(M), as.character(ent$entrez_id))]
M <- M[!is.na(rownames(M)) & !duplicated(rownames(M)), ]

run_voom <- function(samples, grp){
  m <- M[, samples]; g <- factor(grp, levels=c("Early","Late"))
  d <- calcNormFactors(DGEList(m)); d <- d[rowSums(cpm(d)>1) >= 0.10*ncol(d),]
  dm <- model.matrix(~0+g); colnames(dm) <- c("groupEarly","groupLate")
  v <- voom(d, dm); cn <- makeContrasts(Late.Early=groupLate-groupEarly, levels=dm)
  topTable(eBayes(contrasts.fit(lmFit(v,dm),cn)), number=Inf)
}
sig649 <- read.csv("results/ws1_signature/ws1_signature_genes.csv")
# (a) enrichment: rank all tested genes by signed t
full <- read.csv("results/ws1_signature/ws1_dge_full.csv")
rnk <- sort(setNames(full$t, full$GeneSymbol), decreasing=TRUE)
allg <- msigdbr(species="Homo sapiens")
gs <- unique(allg[,c("gs_name","gs_collection")])
paths <- split(allg$gene_symbol, allg$gs_name)
paths <- lapply(paths, unique)
fg <- fgseaMultilevel(paths[intersect(names(paths), unique(allg$gs_name[allg$gs_collection %in% c("H","C2","C5")]))], rnk, minSize=10, maxSize=1000)
fg <- as.data.frame(fg); fg$leadingEdge <- sapply(fg$leadingEdge, paste, collapse=",")
sig_paths <- fg[fg$padj < 0.05,][order(-abs(fg$padj < 0.05)),]
write.csv(fg[order(fg$padj),], "results/ws1_signature/ws8_enrichment_full_ranked.csv", row.names=FALSE)
cat("significant pathways (padj<0.05):", nrow(fg[fg$padj<0.05,]), "\n")
print(head(fg[order(fg$padj), c("pathway","NES","padj","size")], 15))
# (b) sensitivity: exclude paediatric GSE185051
d294 <- disc[disc$cohort != "GSE185051",]
res294 <- run_voom(d294$sample_id, d294$fibrosis_group)
n294 <- sum(res294$adj.P.Val<0.05 & abs(res294$logFC)>0.5)
cat("Sensitivity (exclude paediatric, n=", nrow(d294), "): DEG = ", n294, "\n", sep="")
ov <- length(intersect(rownames(res294)[res294$adj.P.Val<0.05 & abs(res294$logFC)>0.5], sig649$GeneSymbol))
cat("overlap with 649:", ov, "\n")
write.csv(data.frame(metric=c("n","DEG_voom","overlap_with_649"), value=c(nrow(d294), n294, ov)),
          "results/ws1_signature/ws8_sensitivity_excl_paediatric.csv", row.names=FALSE)
# compact panel stability in sensitivity fit
top10 <- c("STMN2","CFAP221","MOXD1","FBLN5","SOX9","PAQR5","SOX9-AS1","ITGBL1","PDZK1IP1","LOXL4")
cat("top-10 in sensitivity DEG set:", sum(top10 %in% rownames(res294)[res294$adj.P.Val<0.05 & abs(res294$logFC)>0.5]), "/10\n")
