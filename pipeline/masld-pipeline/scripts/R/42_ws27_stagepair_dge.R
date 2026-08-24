#!/usr/bin/env Rscript
# WS27 analysis 2.6a — per-adjacent-stage-pair DGE on the locked WS15 discovery matrix.
# Protocol: limma-trend (eBayes trend=TRUE) on the locked log2CPM matrix (12,537 genes x 349
# samples). Deviation from the original ws1 contrast (voom-limma on raw counts) is stated in
# stats_ws27.json: raw counts for the locked universe are not re-derivable here without
# re-running the harmonisation; the locked matrix is the authoritative normalised object.
# Pairs: F0vF1, F1vF2, F2vF3, F3vF4. Output per pair: gene, logFC, t, P.Value, adj.P.Val.

suppressPackageStartupMessages(library(limma))
args <- commandArgs(trailingOnly = TRUE)
mat_csv <- args[1]      # genes x samples log2CPM
stage_csv <- args[2]    # sample_id,stage
out_dir <- args[3]
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

mat <- read.csv(mat_csv, row.names = 1, check.names = FALSE)
st <- read.csv(stage_csv)
rownames(st) <- st$sample_id
st <- st[colnames(mat), ]
stopifnot(!any(is.na(st$stage)))

pairs <- list(c(0, 1), c(1, 2), c(2, 3), c(3, 4))
for (pr in pairs) {
  a <- pr[1]; b <- pr[2]
  keep <- st$stage %in% c(a, b)
  sub <- mat[, keep, drop = FALSE]
  grp <- factor(ifelse(st$stage[keep] == b, "hi", "lo"), levels = c("lo", "hi"))
  design <- model.matrix(~grp)
  fit <- eBayes(lmFit(sub, design), trend = TRUE, robust = TRUE)
  tt <- topTable(fit, number = Inf, sort.by = "none", coef = "grphi")
  out <- data.frame(gene = rownames(tt), logFC = tt$logFC, t = tt$t,
                    P.Value = tt$P.Value, adj.P.Val = tt$adj.P.Val,
                    row.names = NULL)
  fn <- file.path(out_dir, sprintf("stagepair_dge_F%dvF%d.csv", a, b))
  write.csv(out, fn, row.names = FALSE)
  cat(sprintf("F%d vs F%d: n_lo=%d n_hi=%d, genes=%d, significant (adj.P<0.05)=%d\n",
              a, b, sum(grp == "lo"), sum(grp == "hi"), nrow(out),
              sum(out$adj.P.Val < 0.05)))
}
cat("DONE\n")
