# Phase 1 T1/T2: fgsea on limma Late-vs-Early ranked logFC from the 349 (protocol-normalised).
suppressPackageStartupMessages({library(limma); library(fgsea); library(msigdbr)})
setwd("/Users/subhadipbanerjee/masld-revision"); set.seed(42)
OUT <- "results/decisive_test"
D <- read.csv(file.path(OUT,"discovery_qnorm.csv"), row.names=1, check.names=FALSE)
meta <- read.csv("data/discovery_cohort_349.csv"); meta <- meta[match(colnames(D), meta$sample_id),]
grp <- factor(meta$fibrosis_group, levels=c("Early","Late"))
dm <- model.matrix(~0+grp); colnames(dm) <- c("groupEarly","groupLate")
cn <- makeContrasts(Late.Early=groupLate-groupEarly, levels=dm)
fit <- eBayes(contrasts.fit(lmFit(as.matrix(D), dm), cn))
res <- topTable(fit, number=Inf)
rnk <- sort(setNames(res$logFC, rownames(res)), decreasing=TRUE)
write.csv(data.frame(GeneSymbol=names(rnk), logFC=rnk), file.path(OUT,"t1_ranked_logFC.csv"), row.names=FALSE)
deg <- rownames(res)[res$adj.P.Val < 0.05]

sets <- jsonlite::fromJSON(file.path(OUT,"gene_sets.json"), simplifyVector=TRUE)
dr <- sets$drivers; su <- sets$suppressors
pathways <- list(Ferroptosis_Drivers=dr, Ferroptosis_Suppressors=su)

allg <- msigdbr(species="Homo sapiens")
pick <- function(gs) unique(allg$gene_symbol[allg$gs_name==gs & allg$human_gene_symbol!="" | allg$gs_name==gs][1])
comp_names <- c("HALLMARK_APOPTOSIS","GO_NECROPTOTIC_PROCESS","GO_PYROPTOTIC_PROCESS",
                "GO_AUTOPHAGY"= "GO:0006914","HALLMARK_REACTIVE_OXYGEN_SPECIES_PATHWAY","HALLMARK_UNFOLDED_PROTEIN_RESPONSE")
comp <- list(
  HALLMARK_APOPTOSIS = unique(allg$gene_symbol[allg$gs_name=="HALLMARK_APOPTOSIS"]),
  GOBP_NECROPTOTIC_PROCESS = unique(allg$gene_symbol[allg$gs_name=="GOBP_NECROPTOTIC_PROCESS"]),
  REACTOME_PYROPTOSIS = unique(allg$gene_symbol[allg$gs_name=="REACTOME_PYROPTOSIS"]),
  GOBP_MACROAUTOPHAGY = unique(allg$gene_symbol[allg$gs_name=="GOBP_MACROAUTOPHAGY"]),
  HALLMARK_REACTIVE_OXYGEN_SPECIES = unique(allg$gene_symbol[allg$gs_name=="HALLMARK_REACTIVE_OXYGEN_SPECIES_PATHWAY"]),
  HALLMARK_UNFOLDED_PROTEIN_RESPONSE = unique(allg$gene_symbol[allg$gs_name=="HALLMARK_UNFOLDED_PROTEIN_RESPONSE"]))
comp <- lapply(comp, function(g) intersect(g, rownames(D)))
# prefer GO:0006914 autophagy if present as its own term
comp$GOBP_MACROAUTOPHAGY <- intersect(comp$GOBP_MACROAUTOPHAGY, rownames(D))
print(sapply(comp, length))

# 100 random sets matched to driver set on size and mean-expression decile
expr_mean <- rowMeans(as.matrix(D))
bins <- as.integer(cut(rank(expr_mean), breaks=10, labels=FALSE)); names(bins) <- rownames(D)
dr_bin <- table(bins[dr])
rand_sets <- vector("list", 100)
for (i in 1:100) {
  g <- character(0)
  for (b in names(dr_bin)) {
    pool <- rownames(D)[bins==as.integer(b)]; pool <- setdiff(pool, g)
    g <- c(g, sample(pool, min(dr_bin[[b]], length(pool))))
  }
  rand_sets[[i]] <- g
}
names(rand_sets) <- sprintf("RANDOM_%03d", 1:100)

allp <- c(pathways, comp, rand_sets)
fg <- fgseaMultilevel(allp, rnk, minSize=5, maxSize=Inf)
fg <- fg[order(-abs(fg$NES)), ]
fe <- as.data.frame(fg); fe$leadingEdge <- sapply(fg$leadingEdge, paste, collapse=",")
write.csv(fe, file.path(OUT,"t2_fgsea_all.csv"), row.names=FALSE)

ferro <- fe[fe$pathway %in% c("Ferroptosis_Drivers","Ferroptosis_Suppressors"),]
named <- fe[fe$pathway %in% names(comp),]
rand <- fe[grepl("^RANDOM_", fe$pathway),]
cat("\n--- ferroptosis ---\n"); print(ferro[,c("pathway","NES","pval","padj","size")])
cat("\n--- comparators ---\n"); print(named[,c("pathway","NES","pval","padj","size")])
# hypergeometric FerrDb ∩ DEGs (padj<0.05)
uni <- rownames(D)
for (nm in c("drivers","suppressors")) {
  s <- if (nm=="drivers") dr else su
  x <- length(intersect(s, deg)); cat(sprintf("hypergeo %s: overlap=%d/%d, DEG=%d, universe=%d\n", nm, x, length(s), length(deg), length(uni)))
  print(phyper(x-1, length(s), length(uni)-length(s), length(deg), lower.tail=FALSE))
}
saveRDS(list(fg=fe, rand=rand, named=named, ferro=ferro, deg=deg), file.path(OUT,"fgsea_objects.rds"))
