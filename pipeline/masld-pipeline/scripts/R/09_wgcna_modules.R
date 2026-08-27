# WS1.4: proper WGCNA re-run on the 349 (soft-threshold pick, maxBlockSize 20000)
suppressPackageStartupMessages({library(WGCNA); library(dplyr)})
setwd("/Users/subhadipbanerjee/masld-revision"); set.seed(42)
options(stringsAsFactors=FALSE)
OUT <- "results/ws1_signature"
D <- read.csv("results/decisive_test/discovery_qnorm.csv", row.names=1, check.names=FALSE)
datExpr <- t(as.matrix(D))
sft <- pickSoftThreshold(datExpr, powerVector=1:30, networkType="signed", verbose=0)
power <- sft$powerEstimate; cat("soft power:", power, "\n")
net <- blockwiseModules(datExpr, power=power, networkType="signed", TOMType="signed",
                        minModuleSize=30, reassignThreshold=0, mergeCutHeight=0.25,
                        maxBlockSize=20000, numericLabels=TRUE, pamRespectsDendro=FALSE, verbose=0)
mods <- labels2colors(net$colors)
cat("genes:", nrow(datExpr), " modules:", length(unique(mods)), "\n"); print(table(mods))
disc <- read.csv("data/discovery_cohort_349.csv")
stage <- disc$fibrosis_stage[match(colnames(D), disc$sample_id)]
MEs <- orderMEs(moduleEigengenes(datExpr, mods)$eigengenes)
corSt <- as.data.frame(cor(MEs, stage, use="p"))
pSt <- as.data.frame(corPvalueStudent(as.numeric(corSt[,1]), nrow(datExpr)))
res <- data.frame(module=sub("^ME","",rownames(corSt)), cor_stage=corSt[,1], p=pSt[,1], size=as.vector(table(mods)[sub("^ME","",rownames(corSt))]))
res <- res[order(-abs(res$cor_stage)),]
print(head(res, 8))
write.csv(res, file.path(OUT,"ws1_wgcna_module_stage.csv"), row.names=FALSE)
top <- res$module[1]
genes <- colnames(D)[mods==sub("^ME","",top)]; top <- sub("^ME","",top)
writeLines(genes, file.path(OUT, sprintf("ws1_wgcna_top_module_%s_genes.txt", top)))
jsonlite::write_json(list(soft_power=power, n_genes=nrow(datExpr), n_modules=length(unique(mods)),
  module_table=res, maxBlockSize=20000, top_module=top, top_module_size=length(genes)),
  file.path(OUT,"stats_ws1_wgcna.json"), auto_unbox=TRUE, digits=NA)
