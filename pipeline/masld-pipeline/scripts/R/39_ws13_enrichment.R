suppressPackageStartupMessages({library(msigdbr)})
setwd("/Users/subhadipbanerjee/masld-revision")
bg <- readLines("results/ws13/_bg.txt")
pv <- read.delim("results/ws13/_panels.txt", sep="\t", header=FALSE, col.names=c("set","genes"))
mods <- read.delim("results/ws13/_modules.txt", sep="\t", header=FALSE, col.names=c("set","genes"))
sets <- rbind(pv, mods)
allg <- msigdbr(species="Homo sapiens")
getset <- function(nm) unique(allg$gene_symbol[allg$gs_name==nm])
ductural_curated <- c("KRT7","KRT19","EPCAM","SOX9","HNF1B","CFTR","ONECUT1")
curated_ductular <- c("KRT7","KRT19","EPCAM","SOX9","HNF1B","CFTR","ONECUT1")
targets <- list(
  NABA_MATRISOME=getset("NABA_MATRISOME"),
  NABA_CORE_MATRISOME=getset("NABA_CORE_MATRISOME"),
  NABA_ECM_REGULATORS=getset("NABA_ECM_REGULATORS"),
  NABA_ECM_AFFILIATED=getset("NABA_ECM_AFFILIATED"),
  GO_ELASTIC_FIBRE_ASSEMBLY=getset("GOBP_ELASTIC_FIBER_ASSEMBLY"),
  GO_ECM_ORGANISATION=getset("GOBP_EXTRACELLULAR_MATRIX_ORGANIZATION"),
  DUCTULAR_CURATED=curated_ductular)
N <- length(bg)
out <- data.frame()
for (i in seq_len(nrow(sets))) {
  genes <- unlist(strsplit(sets$genes[i], ","))
  inset <- intersect(genes, bg)
  K <- length(inset)
  for (nm in names(targets)) {
    tgt <- intersect(targets[[nm]], bg)
    ov <- intersect(inset, tgt)
    p <- phyper(length(ov)-1, length(tgt), N-length(tgt), K, lower.tail=FALSE)
    out <- rbind(out, data.frame(set=sets$set[i], annotation=nm, n_set=K, n_annotation=length(tgt),
      overlap=length(ov), overlapping_genes=paste(ov, collapse=";"), p=p))
  }
}
write.csv(out, "results/ws13/module_enrichment.csv", row.names=FALSE)
print(out[, c("set","annotation","n_set","overlap","p")])
