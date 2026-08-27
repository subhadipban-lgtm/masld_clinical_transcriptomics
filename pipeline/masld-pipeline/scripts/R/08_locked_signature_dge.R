# WS1: locked pre-specified signature pipeline.
# DGE: voom + lmFit + eBayes on raw counts, ~ fibrosis_group (no age; unavailable 216/349).
# Threshold: padj < 0.05 AND |log2FC| > 0.5. Signature = whatever this yields.
suppressPackageStartupMessages({library(limma); library(edgeR)})
setwd("/Users/subhadipbanerjee/masld-revision"); set.seed(42)
OUT <- "results/ws1_signature"; dir.create(OUT, recursive=TRUE, showWarnings=FALSE)

disc <- read.csv("data/discovery_cohort_349.csv", stringsAsFactors=FALSE)
ent <- read.delim("data/kamzolas/gene_symbols_to_entrez.tsv", stringsAsFactors=FALSE)
ent <- ent[!is.na(ent$entrez_id) & !is.na(ent$gene_symbol) & !duplicated(ent$entrez_id),]
gses <- c("GSE135251","GSE130970","GSE185051")
cl <- lapply(gses, function(g){ f <- list.files("zenodo_upload/raw_data", pattern=paste0("^",g), full.names=TRUE)
  read.delim(gzfile(f), sep="\t", header=TRUE, row.names=1, check.names=FALSE, stringsAsFactors=FALSE)})
common <- Reduce(intersect, lapply(cl, rownames))
M <- as.matrix(do.call(cbind, lapply(cl, function(d) as.matrix(d[common,])))); storage.mode(M) <- "numeric"
M <- M[, disc$sample_id]
rownames(M) <- ent$gene_symbol[match(rownames(M), as.character(ent$entrez_id))]
M <- M[!is.na(rownames(M)) & !duplicated(rownames(M)), ]
cat("count matrix:", nrow(M), "genes x", ncol(M), "samples\n")

grp <- factor(disc$fibrosis_group, levels=c("Early","Late"))
d <- DGEList(M); d <- calcNormFactors(d)
keep <- rowSums(cpm(d) > 1) >= 0.10 * ncol(d)
d <- d[keep,]; cat("after CPM>1 in >=10%:", nrow(d), "genes\n")
dm <- model.matrix(~0+grp); colnames(dm) <- c("groupEarly","groupLate")
v <- voom(d, dm)
cn <- makeContrasts(Late.Early=groupLate-groupEarly, levels=dm)
fit <- eBayes(contrasts.fit(lmFit(v, dm), cn))
res <- topTable(fit, number=Inf)
res$GeneSymbol <- rownames(res)
sig <- res[res$adj.P.Val < 0.05 & abs(res$logFC) > 0.5, ]
cat("LOCKED SIGNATURE SIZE:", nrow(sig), "\n")
cat("up:", sum(sig$logFC>0), " down:", sum(sig$logFC<0), "\n")
cat("top 20 by padj:\n"); print(head(sig[order(sig$adj.P.Val), c("GeneSymbol","logFC","adj.P.Val")], 20))
write.csv(res, file.path(OUT,"ws1_dge_full.csv"), row.names=FALSE)
write.csv(sig, file.path(OUT,"ws1_signature_genes.csv"), row.names=FALSE)
js <- list(n_samples=nrow(disc), design="~ fibrosis_group (voom, no age covariate)",
  genes_tested=nrow(res), threshold="padj<0.05 & |log2FC|>0.5",
  signature_size=nrow(sig), n_up=sum(sig$logFC>0), n_down=sum(sig$logFC<0),
  n_deg_padj_only=sum(res$adj.P.Val<0.05),
  top20_by_padj=head(sig[order(sig$adj.P.Val), c("GeneSymbol","logFC","adj.P.Val")], 20))
jsonlite::write_json(js, file.path(OUT,"stats_ws1_signature.json"), auto_unbox=TRUE, digits=NA)
