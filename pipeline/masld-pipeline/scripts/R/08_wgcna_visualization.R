# --- Publication-Grade WGCNA Figure (Revised Figure 8) -----------------------
# Panels:
#   8A – Fibrosis-module GO enrichment (ECM/EMT/inflammation-focused)
#   8B – Hub-only view of the fibrosis module
#   8C – MEblue vs ferroptosis suppressor GSVA (color = stage, size = age)

message("Setting up libraries...")

# 0. Ensure BiocManager and WGCNA dependencies ------------------------------

if (!requireNamespace("BiocManager", quietly = TRUE)) {
  install.packages("BiocManager")
}

# install impute + preprocessCore only if missing
needed_bioc <- c("impute", "preprocessCore")
missing_bioc <- needed_bioc[!needed_bioc %in% installed.packages()[, "Package"]]
if (length(missing_bioc) > 0) {
  BiocManager::install(missing_bioc)
}

# load WGCNA core dependencies
library(impute)
library(preprocessCore)

# 1. CRAN / Bioc packages used in the figure --------------------------------

pkgs <- c(
  "tidyverse", "WGCNA", "clusterProfiler", "org.Hs.eg.db",
  "igraph", "ggraph", "ggrepel", "patchwork"
)

new_pkgs <- pkgs[!(pkgs %in% installed.packages()[, "Package"])]
if (length(new_pkgs)) {
  message("Installing missing packages: ", paste(new_pkgs, collapse = ", "))
  if (any(new_pkgs %in% c("clusterProfiler", "org.Hs.eg.db"))) {
    BiocManager::install(intersect(new_pkgs, c("clusterProfiler", "org.Hs.eg.db")))
  }
  cran_pkgs <- setdiff(new_pkgs, c("clusterProfiler", "org.Hs.eg.db"))
  if (length(cran_pkgs)) install.packages(cran_pkgs)
}

suppressPackageStartupMessages({
  library(tidyverse)
  library(WGCNA)
  library(clusterProfiler)
  library(org.Hs.eg.db)
  library(igraph)
  library(ggraph)
  library(ggrepel)
  library(patchwork)
})

# ---------------------------------------------------------------------------
# 1. Data loading
# ---------------------------------------------------------------------------
message("Loading data...")

required_files <- c(
  "WGCNA_fibrosis_module_genes.csv",
  "harmonized_ferroptosis_GSVA_scores.csv",
  "normalized_expression_matrix.rds",
  "harmonized_MASLD_metadata.csv"
)

if (!all(file.exists(required_files))) {
  stop(
    "Missing files: ",
    paste(required_files[!file.exists(required_files)], collapse = ", ")
  )
}

module_genes_df <- read.csv("WGCNA_fibrosis_module_genes.csv")
fibrosis_genes  <- module_genes_df$genes   # change if your column name differs

ferro_scores_raw <- read.csv(
  "harmonized_ferroptosis_GSVA_scores.csv",
  check.names = FALSE
)

expr_matrix <- readRDS("normalized_expression_matrix.rds")

metadata_raw <- read.csv(
  "harmonized_MASLD_metadata.csv",
  check.names = FALSE
)

# ---------------------------------------------------------------------------
# 2. Harmonize sample IDs to `sample_id`
# ---------------------------------------------------------------------------

harmonize_ids <- function(df, candidates = c("SampleID", "sample_id", "sample", "Sample")) {
  present <- intersect(candidates, names(df))
  if (length(present) == 0) {
    stop("Could not find a sample ID column in: ", deparse(substitute(df)),
         ". Tried: ", paste(candidates, collapse = ", "))
  }
  dplyr::rename(df, sample_id = !!rlang::sym(present[1]))
}

ferro_scores <- harmonize_ids(ferro_scores_raw)
metadata     <- harmonize_ids(metadata_raw)

# ---------------------------------------------------------------------------
# 3. Prepare expression matrix in WGCNA format
# ---------------------------------------------------------------------------

datExpr <- as.data.frame(t(expr_matrix))   # samples in rows, genes in columns

# ---------------------------------------------------------------------------
# 4. Panel 8A – GO enrichment (unchanged logic)
# ---------------------------------------------------------------------------

message("Panel 8A: GO enrichment for fibrosis module...")

entrez_ids <- bitr(
  fibrosis_genes,
  fromType = "SYMBOL", toType = "ENTREZID",
  OrgDb   = "org.Hs.eg.db", drop = TRUE
)

go_res <- enrichGO(
  gene          = entrez_ids$ENTREZID,
  OrgDb         = org.Hs.eg.db,
  keyType       = "ENTREZID",
  ont           = "BP",
  pAdjustMethod = "BH",
  pvalueCutoff  = 0.05,
  qvalueCutoff  = 0.2
)

if (!is.null(go_res) && nrow(go_res) > 0) {
  go_tbl <- as.data.frame(go_res) %>%
    dplyr::mutate(
      Description = stringr::str_wrap(Description, width = 40),
      GeneRatio   = purrr::map_dbl(GeneRatio, ~ {
        x <- strsplit(.x, "/")[[1]]
        as.numeric(x[1]) / as.numeric(x[2])
      })
    ) %>%
    dplyr::filter(
      stringr::str_detect(tolower(Description),
                          "extracellular matrix|collagen|fibril|ecm|adhesion|mesenchymal|wound|fibroblast|inflamm")
      | dplyr::row_number() <= 15
    ) %>%
    dplyr::slice_min(order_by = p.adjust, n = 10)

  plot_a <- ggplot(go_tbl,
                   aes(x = GeneRatio,
                       y = forcats::fct_reorder(Description, GeneRatio))) +
    geom_point(aes(size = Count, colour = -log10(p.adjust)),
               stroke = 0.2, alpha = 0.9) +
    scale_colour_viridis_c(name = "-log10 adj. P") +
    scale_size(range = c(1.8, 5.5), name = "Gene count") +
    labs(
      title = "Fibrosis module – GO biological processes",
      x = "Gene ratio",
      y = NULL
    ) +
    theme_classic(base_size = 10) +
    theme(
      plot.title = element_text(hjust = 0.5, face = "bold"),
      axis.text.y = element_text(size = 8)
    )
} else {
  plot_a <- ggplot() +
    annotate("text", x = 0, y = 0, label = "No significant GO terms",
             size = 3) +
    theme_void()
}


# ---------------------------------------------------------------------------
# 3. Panel 8B – Hub-only view of fibrosis module
# ---------------------------------------------------------------------------
message("Panel 8B: Hub-only fibrosis-module network...")

# adjacency from TOM blocks (same logic, but we keep only strongest hubs)
tom_files <- list.files(pattern = "masldTOM-block.*\\.RData$")
if (length(tom_files) == 0) {
  warning("No TOM files found – Panel 8B will be skipped.")
  plot_b <- ggplot() + theme_void() +
    annotate("text", x = 0, y = 0, label = "TOM files not available", size = 3)
} else {
  # build adjacency for module genes
  adj_sub <- matrix(
    0,
    nrow = length(fibrosis_genes), ncol = length(fibrosis_genes),
    dimnames = list(fibrosis_genes, fibrosis_genes)
  )

  for (f in tom_files) {
    message("  -> integrating ", f)
    load(f)  # provides TOM
    genes_block <- colnames(TOM)
    mg_block    <- intersect(fibrosis_genes, genes_block)
    if (length(mg_block) > 0) {
      adj_sub[mg_block, mg_block] <- as.matrix(TOM[mg_block, mg_block])
    }
    rm(TOM); gc()
  }

  # restrict to top edges to avoid hairball
  n_top <- 1000
  thr   <- sort(adj_sub, decreasing = TRUE)[n_top]
  adj_sub[adj_sub < thr] <- 0

  g <- graph_from_adjacency_matrix(adj_sub,
                                   mode = "undirected",
                                   weighted = TRUE,
                                   diag = FALSE)

  # intramodular connectivity
  conn <- strength(g)
  V(g)$connectivity <- conn

  # keep only top 25 hubs as nodes
  keep_genes <- names(sort(conn, decreasing = TRUE))[1:25]
  g_hub <- induced_subgraph(g, vids = keep_genes)

  # prepare layout and labels
  V(g_hub)$is_hub <- TRUE
  V(g_hub)$label  <- V(g_hub)$name

  set.seed(123)
  plot_b <- ggraph(g_hub, layout = "fr") +
    geom_edge_link(aes(alpha = weight),
                   colour = "grey80", show.legend = FALSE) +
    geom_node_point(aes(size = connectivity),
                    colour = "#1f78b4", alpha = 0.9) +
    geom_node_text(aes(label = label),
                   size = 3, fontface = "bold",
                   repel = TRUE) +
    scale_size_continuous(range = c(2, 8), name = "Connectivity") +
    theme_graph(base_family = "sans") +
    labs(title = "Top hub genes in the fibrosis module") +
    theme(plot.title = element_text(hjust = 0.5, face = "bold", size = 10))
}

# ---------------------------------------------------------------------------
# 4. Panel 8C – MEblue vs ferroptosis suppressor GSVA (age-aware)
# ---------------------------------------------------------------------------
message("Panel 8C: MEblue vs ferroptosis suppressor GSVA...")

moduleColors <- ifelse(colnames(datExpr) %in% fibrosis_genes, "blue", "grey")

MEs <- moduleEigengenes(datExpr, colors = moduleColors)$eigengenes
fibrosis_ME <- MEs %>%
  dplyr::select(MEblue) %>%
  tibble::rownames_to_column("sample_id")

meta_small <- metadata %>%
  dplyr::select(sample_id, fibrosis_stage, age)

plot_data <- fibrosis_ME %>%
  dplyr::inner_join(ferro_scores, by = "sample_id") %>%
  dplyr::inner_join(meta_small,  by = "sample_id")

suppressor_col <- "GSVA_Suppressors_Score"  # adjust if needed
stopifnot(suppressor_col %in% colnames(plot_data))

corr <- cor.test(
  plot_data$MEblue,
  plot_data[[suppressor_col]],
  method = "pearson"
)
corr_text <- sprintf("R = %.2f\nP = %.1e",
                     corr$estimate, corr$p.value)

size_scaled <- scales::rescale(plot_data$age, to = c(1.5, 5))

plot_c <- ggplot(
  plot_data,
  aes(x = MEblue, y = .data[[suppressor_col]])
) +
  geom_point(
    aes(colour = as.factor(fibrosis_stage),
        size   = size_scaled),
    alpha = 0.7, stroke = 0.2
  ) +
  geom_smooth(method = "lm", se = FALSE, colour = "black", linewidth = 0.6) +
  scale_colour_viridis_d(name = "Fibrosis stage") +
  scale_size_identity(guide = "none") +
  labs(
    title = "Fibrosis module eigengene vs ferroptosis suppressor activity",
    x = "Fibrosis module eigengene (MEblue)",
    y = "Ferroptosis suppressor score (GSVA)"
  ) +
  annotate(
    "text",
    x = min(plot_data$MEblue, na.rm = TRUE),
    y = max(plot_data[[suppressor_col]], na.rm = TRUE),
    hjust = 0, vjust = 1,
    label = corr_text,
    size = 3,
    fontface = "bold"
  ) +
  theme_classic(base_size = 10) +
  theme(
    plot.title = element_text(hjust = 0.5, face = "bold"),
    legend.position = "right"
  )
ggsave("Figure8_WGCNA_Fibrosis_Ferroptosis_Publication.png",
       final_plot, width = 10, height = 7, dpi = 300)

message("\n✅ Figure 8 panels generated and saved as 'Figure8_WGCNA_Fibrosis_Ferroptosis_Publication.png'")
