# --- Comprehensive Analysis of Ferroptosis in Liver Fibrosis ---
# This script performs differential gene expression, gene set enrichment,
# and generates publication-grade visualizations to explore the role of
# ferroptosis drivers and suppressors in MASLD fibrosis progression.

# --- 1. SETUP: Install and Load Libraries ---
message("--- 1. Installing and Loading Libraries ---")

# A helper function to install packages if they are not already present
install_and_load <- function(packages) {
  for (pkg in packages) {
    if (!require(pkg, character.only = TRUE)) {
      message(paste("Installing", pkg))
      if (pkg %in% c("limma", "clusterProfiler", "org.Hs.eg.db", "enrichplot")) {
        BiocManager::install(pkg)
      } else {
        install.packages(pkg)
      }
    }
    library(pkg, character.only = TRUE)
  }
}

required_packages <- c(
  "tidyverse", "limma", "pheatmap", "RColorBrewer", "ggrepel",
  "clusterProfiler", "org.Hs.eg.db", "enrichplot", "ggpubr", "corrplot"
)

install_and_load(required_packages)

# --- 2. DATA PREPARATION ---
message("\n--- 2. Loading and Pre-processing Data ---")

# Load datasets
metadata <- read.csv("harmonized_MASLD_metadata.csv")
norm_expr_matrix <- readRDS("normalized_expression_matrix.rds")
ferroptosis_drivers <- read.csv("ferroptosis_driver.csv")
ferroptosis_suppressors <- read.csv("ferroptosis_suppressor.csv")

# Clean metadata and define fibrosis groups
meta_filtered <- metadata %>%
  filter(!is.na(fibrosis_stage) & !is.na(age)) %>%
  mutate(
    fibrosis_group = case_when(
      fibrosis_stage <= 2 ~ "Early",
      fibrosis_stage >= 3 ~ "Late"
    )
  ) %>%
  filter(fibrosis_group %in% c("Early", "Late"))

# **FIX**: Robustly align expression matrix and metadata
# Find the common samples that exist in BOTH files
common_samples <- intersect(meta_filtered$sample_id, colnames(norm_expr_matrix))

message(paste("Found", length(common_samples), "common samples for analysis."))

# Filter both the expression matrix and metadata to only keep common samples
expr_aligned <- norm_expr_matrix[, common_samples]
meta_clean <- meta_filtered %>%
  filter(sample_id %in% common_samples) %>%
  # Arrange metadata in the same order as the expression matrix columns for perfect alignment
  arrange(match(sample_id, colnames(expr_aligned))) %>%
  column_to_rownames("sample_id") # Set sample_id as rownames for easier matching in limma/pheatmap


# Prepare ferroptosis gene lists
driver_genes <- unique(ferroptosis_drivers$symbol)
suppressor_genes <- unique(ferroptosis_suppressors$symbol)
all_ferroptosis_genes <- unique(c(driver_genes, suppressor_genes))


# --- 3. DIFFERENTIAL GENE EXPRESSION ANALYSIS (DGEA) ---
message("\n--- 3. Performing Differential Gene Expression Analysis with limma ---")

# Create the design matrix for limma
design <- model.matrix(~ 0 + fibrosis_group + age, data = meta_clean)
colnames(design) <- c("Early", "Late", "Age") # Make column names syntactically valid

# Define the contrast: we want to find genes different between Late and Early fibrosis
contrast_matrix <- makeContrasts(Late_vs_Early = Late - Early, levels = design)

# Fit the linear model
fit <- lmFit(expr_aligned, design)
fit_contrast <- contrasts.fit(fit, contrast_matrix)
fit_bayes <- eBayes(fit_contrast)

# Get the full table of results
dge_results <- topTable(fit_bayes, number = Inf, coef = "Late_vs_Early") %>%
  rownames_to_column("GeneSymbol")

message(paste("DGEA complete. Found", nrow(dge_results), "genes."))

# --- 4. GENE SET ENRICHMENT ANALYSIS (GSEA) ---
message("\n--- 4. Performing Gene Set Enrichment Analysis for Ferroptosis ---")

# Prepare a ranked list of genes for GSEA (ranked by t-statistic)
ranked_genes <- dge_results %>%
  filter(!is.na(t)) %>% # Ensure no NA values in the ranking metric
  select(GeneSymbol, t) %>%
  deframe() %>%
  sort(decreasing = TRUE)

# Create the ferroptosis gene set for GSEA
ferroptosis_term <- data.frame(
  term = "Ferroptosis",
  gene = all_ferroptosis_genes
)

# Run GSEA
gsea_result <- GSEA(ranked_genes, TERM2GENE = ferroptosis_term, pvalueCutoff = 1.0, verbose = FALSE)

# Visualize GSEA result
if (nrow(gsea_result@result) > 0) {
  gsea_plot <- gseaplot2(gsea_result, geneSetID = 1, title = "Enrichment of Ferroptosis Gene Set")
  ggsave("GSEA_Ferroptosis_Enrichment.png", gsea_plot, width = 8, height = 6, dpi = 300)
  message("GSEA plot saved to GSEA_Ferroptosis_Enrichment.png")
} else {
  message("GSEA did not yield any significant results for the ferroptosis gene set.")
}


# --- 5. PUBLICATION-GRADE VISUALIZATIONS ---
message("\n--- 5. Generating Publication-Grade Visualizations ---")

# -- Visualization 5A: Heatmap of Differentially Expressed Ferroptosis Genes --
message("-> Generating Heatmap...")

# Identify significant DEGs that are also in our ferroptosis list
significant_ferroptosis_degs <- dge_results %>%
  filter(adj.P.Val < 0.05 & GeneSymbol %in% all_ferroptosis_genes) %>%
  # Add a column for the gene's role
  mutate(Role = case_when(
    GeneSymbol %in% driver_genes ~ "Driver",
    GeneSymbol %in% suppressor_genes ~ "Suppressor"
  ))

if (nrow(significant_ferroptosis_degs) > 5) { # Check if we have enough genes for a meaningful heatmap
  # Prepare data for the heatmap
  heatmap_data <- expr_aligned[significant_ferroptosis_degs$GeneSymbol, ]

  # Create annotations for heatmap columns (samples) and rows (genes)
  annotation_col <- meta_clean[, c("fibrosis_group", "fibrosis_stage", "age")]
  annotation_row <- significant_ferroptosis_degs %>%
    select(GeneSymbol, Role) %>%
    column_to_rownames("GeneSymbol")

  # Define colors
  ann_colors <- list(
    fibrosis_group = c(Early = "#0072B2", Late = "#D55E00"),
    Role = c(Driver = "firebrick", Suppressor = "forestgreen"),
    fibrosis_stage = colorRampPalette(c("lightblue", "darkblue"))(length(unique(annotation_col$fibrosis_stage))),
    age = colorRampPalette(c("white", "purple"))(100)
  )

  # Generate heatmap
  pheatmap(
    heatmap_data,
    main = "Expression of Significant Ferroptosis-Related DEGs",
    annotation_col = annotation_col,
    annotation_row = annotation_row,
    annotation_colors = ann_colors,
    scale = "row", # Z-score scaling by row (gene)
    show_colnames = FALSE,
    cluster_cols = TRUE,
    cutree_cols = 2,
    filename = "Heatmap_Ferroptosis_DEGs.png",
    width = 12, height = 10
  )
  message("Heatmap saved to Heatmap_Ferroptosis_DEGs.png")
} else {
  message("Skipping Heatmap: Fewer than 5 significant ferroptosis-related DEGs found.")
}


# -- Visualization 5B: Co-expression Heatmap (Corrplot) --
if (nrow(significant_ferroptosis_degs) > 5) {
  message("-> Generating Co-expression Plot...")

  # Calculate correlation matrix for the significant ferroptosis genes
  # Ensure heatmap_data exists from the previous step
  if(exists("heatmap_data")){
    cor_matrix <- cor(t(heatmap_data)) # Transpose because cor() works on columns

    png("Coexpression_Corrplot_Ferroptosis.png", width = 10, height = 10, units = "in", res = 300)
    corrplot(
      cor_matrix,
      method = "color",
      order = "hclust", # Hierarchically cluster genes
      tl.col = "black", tl.cex = 0.7,
      addCoef.col = "black", number.cex = 0.5,
      title = "Co-expression of Ferroptosis-Related DEGs",
      mar = c(0, 0, 1, 0) # Adjust margins
    )
    dev.off()
    message("Co-expression plot saved to Coexpression_Corrplot_Ferroptosis.png")
  }
} else {
  message("Skipping Co-expression plot: Not enough significant genes.")
}


# -- Visualization 5C: Volcano Plot Highlighting Ferroptosis Genes --
message("-> Generating Volcano Plot...")

volcano_plot_data <- dge_results %>%
  mutate(
    Highlight = case_when(
      adj.P.Val < 0.05 & GeneSymbol %in% driver_genes ~ "Driver",
      adj.P.Val < 0.05 & GeneSymbol %in% suppressor_genes ~ "Suppressor",
      adj.P.Val < 0.05 ~ "Significant",
      TRUE ~ "Not Significant"
    ),
    Label = if_else(Highlight %in% c("Driver", "Suppressor"), GeneSymbol, "")
  )

volcano_plot <- ggplot(volcano_plot_data, aes(x = logFC, y = -log10(adj.P.Val))) +
  geom_point(aes(color = Highlight), alpha = 0.7) +
  geom_text_repel(aes(label = Label), max.overlaps = 15, size = 3) +
  scale_color_manual(values = c(
    "Driver" = "firebrick", "Suppressor" = "forestgreen",
    "Significant" = "grey50", "Not Significant" = "grey80"
  )) +
  labs(
    title = "Volcano Plot: Late vs. Early Fibrosis",
    x = "Log2 Fold Change",
    y = "-log10(Adjusted P-value)"
  ) +
  theme_minimal(base_size = 14) +
  geom_vline(xintercept = 0, linetype = "dashed") +
  geom_hline(yintercept = -log10(0.05), linetype = "dashed")

ggsave("Volcano_Plot_Ferroptosis.png", volcano_plot, width = 10, height = 8, dpi = 300)
message("Volcano plot saved to Volcano_Plot_Ferroptosis.png")


# -- Visualization 5D: Density Plot of LogFC for Drivers vs. Suppressors --
message("-> Generating Density Plot...")

density_data <- dge_results %>%
  filter(GeneSymbol %in% all_ferroptosis_genes) %>%
  mutate(Role = ifelse(GeneSymbol %in% driver_genes, "Driver", "Suppressor"))

density_plot <- ggplot(density_data, aes(x = logFC, fill = Role)) +
  geom_density(alpha = 0.6) +
  scale_fill_manual(values = c("Driver" = "firebrick", "Suppressor" = "forestgreen")) +
  labs(
    title = "Distribution of Fold Changes for Ferroptosis Genes",
    x = "Log2 Fold Change (Late vs. Early Fibrosis)",
    y = "Density"
  ) +
  theme_minimal(base_size = 14) +
  geom_vline(xintercept = 0, linetype = "dashed")

ggsave("Density_Plot_LogFC_Ferroptosis.png", density_plot, width = 8, height = 6, dpi = 300)
message("Density plot saved to Density_Plot_LogFC_Ferroptosis.png")


# -- Visualization 5E: Scatter Plot of a Key Driver vs. Suppressor --
message("-> Generating Scatter Plot...")

# Select the most significant driver and suppressor for plotting
top_driver <- significant_ferroptosis_degs %>% filter(Role == "Driver") %>% slice_min(adj.P.Val, n = 1) %>% pull(GeneSymbol)
top_suppressor <- significant_ferroptosis_degs %>% filter(Role == "Suppressor") %>% slice_min(adj.P.Val, n = 1) %>% pull(GeneSymbol)

if (length(top_driver) > 0 && length(top_suppressor) > 0) {
  scatter_data <- t(expr_aligned[c(top_driver, top_suppressor), ]) %>%
    as.data.frame() %>%
    rownames_to_column("sample_id") %>%
    left_join(rownames_to_column(meta_clean, "sample_id"), by = "sample_id")

  scatter_plot <- ggscatter(
    scatter_data,
    x = top_driver, y = top_suppressor,
    color = "fibrosis_group",
    add = "reg.line",  # Add regression line
    conf.int = TRUE,       # Add confidence interval
    cor.coef = TRUE,       # Add correlation coefficient
    cor.method = "pearson"
  ) +
    labs(
      title = paste("Co-expression of", top_driver, "(Driver) vs.", top_suppressor, "(Suppressor)"),
      x = paste(top_driver, "Expression (log2 CPM)"),
      y = paste(top_suppressor, "Expression (log2 CPM)")
    ) +
    scale_color_manual(values = c(Early = "#0072B2", Late = "#D55E00"))

  ggsave("Scatter_Plot_Driver_vs_Suppressor.png", scatter_plot, width = 8, height = 7, dpi = 300)
  message(paste("Scatter plot saved to Scatter_Plot_Driver_vs_Suppressor.png, comparing", top_driver, "and", top_suppressor))
} else {
  message("Skipping scatter plot: could not find a significant driver and/or suppressor.")
}

message("\n--- ANALYSIS COMPLETE ---")

