#!/usr/bin/env Rscript
# WS30 Task 1 — per-adjacent-stage-pair DGE on the LOCKED WS15 Fujiwara matrix,
# identical protocol to scripts/R/42_ws27_stagepair_dge.R (limma-trend, trend=TRUE, robust).
suppressPackageStartupMessages(library(limma))
args <- commandArgs(trailingOnly = TRUE)
mat_csv <- args[1]; stage_csv <- args[2]; out_dir <- args[3]
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
mat <- read.csv(mat_csv, row.names = 1, check.names = FALSE)
st <- read.csv(stage_csv); rownames(st) <- st$sample_id; st <- st[colnames(mat), ]
stopifnot(!any(is.na(st$stage)))
for (pr in list(c(0, 1), c(1, 2), c(2, 3), c(3, 4))) {
  a <- pr[1]; b <- pr[2]
  keep <- st$stage %in% c(a, b)
  sub <- mat[, keep, drop = FALSE]
  grp <- factor(ifelse(st$stage[keep] == b, "hi", "lo"), levels = c("lo", "hi"))
  fit <- eBayes(lmFit(sub, model.matrix(~grp)), trend = TRUE, robust = TRUE)
  tt <- topTable(fit, number = Inf, sort.by = "none", coef = "grphi")
  write.csv(data.frame(gene = rownames(tt), logFC = tt$logFC, t = tt$t,
                       P.Value = tt$P.Value, adj.P.Val = tt$adj.P.Val),
            file.path(out_dir, sprintf("fuji_stagepair_dge_F%dvF%d.csv", a, b)), row.names = FALSE)
  cat(sprintf("F%d vs F%d: n_lo=%d n_hi=%d genes=%d sig=%d\n", a, b,
              sum(grp == "lo"), sum(grp == "hi"), nrow(tt), sum(tt$adj.P.Val < 0.05)))
}
cat("DONE\n")
