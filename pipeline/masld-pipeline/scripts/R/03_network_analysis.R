# --- Unified Pipeline: From DGE to Advanced Network and Pathway Analysis ---
# This comprehensive script performs a complete analysis workflow:
# 1. Loads and prepares raw expression data and metadata.
# 2. Conducts differential gene expression (DGE) analysis.
# 3. Creates a complex volcano plot showing significance for multiple factors.
# 4. Uses DGE results for comprehensive downstream analysis, including:
#    - Multi-database pathway and gene ontology enrichment (GO, KEGG, Reactome, MSigDB).
#    - Advanced visualizations (Bubble Plots, Enrichment Maps, GOCircle).
#    - Construction of a protein-protein interaction (PPI) network.
#    - Network visualization (Force-directed layout, Heatmap).
#    - Export of final results to CSV and network files (including .graphml and .sif).

# --- 1. SETUP: Install and Load All Necessary Libraries ---
message("--- 1. Installing and Loading All Libraries ---")

# A helper function to install packages if they are not already present
install_and_load <- function(packages) {
  for (pkg in packages) {
    if (!require(pkg, character.only = TRUE)) {
      message(paste("Installing", pkg))
      if (pkg %in% c("limma", "clusterProfiler", "org.Hs.eg.db", "ReactomePA", "enrichplot", "STRINGdb", "GOplot")) {
        if (!require("BiocManager", quietly = TRUE)) install.packages("BiocManager")
        BiocManager::install(pkg)
      } else {
        install.packages(pkg)
      }
    }
    library(pkg, character.only = TRUE)
  }
}

required_packages <- c(
  "tidyverse", "limma", "pheatmap", "RColorBrewer", "ggrepel", "clusterProfiler",
  "org.Hs.eg.db", "ReactomePA", "enrichplot", "msigdbr", "igraph", "ggraph",
  "STRINGdb", "GOplot"
)

install_and_load(required_packages)

# --- 2. DATA PREPARATION & OUTPUT DIRECTORIES ---
message("\n--- 2. Loading, Pre-processing Data, and Setting up Output Directories ---")

# Create main output directories
if (!dir.exists("unified_analysis_outputs")) dir.create("unified_analysis_outputs")
if (!dir.exists("unified_analysis_outputs/enrichment_results")) dir.create("unified_analysis_outputs/enrichment_results")
if (!dir.exists("unified_analysis_outputs/figures")) dir.create("unified_analysis_outputs/figures")
if (!dir.exists("unified_analysis_outputs/networks")) dir.create("unified_analysis_outputs/networks")

# Check for input files
if (!file.exists("harmonized_MASLD_metadata.csv") || !file.exists("normalized_expression_matrix.rds")) {
  stop("FATAL ERROR: Input files 'harmonized_MASLD_metadata.csv' and/or 'normalized_expression_matrix.rds' not found.")
}

# Load datasets
metadata <- read.csv("harmonized_MASLD_metadata.csv")
norm_expr_matrix <- readRDS("normalized_expression_matrix.rds")

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

# Robustly align expression matrix and metadata
common_samples <- intersect(meta_filtered$sample_id, colnames(norm_expr_matrix))
message(paste("Found", length(common_samples), "common samples for analysis."))

expr_aligned <- norm_expr_matrix[, common_samples]
meta_clean <- meta_filtered %>%
  filter(sample_id %in% common_samples) %>%
  arrange(match(sample_id, colnames(expr_aligned))) %>%
  column_to_rownames("sample_id")

# --- 3. DIFFERENTIAL GENE EXPRESSION ANALYSIS (DGEA) ---
message("\n--- 3. Performing Differential Gene Expression Analysis with limma ---")

design <- model.matrix(~ 0 + fibrosis_group + age, data = meta_clean)
colnames(design) <- c("Early", "Late", "Age")

# FIX: Include 'Age' in the contrast matrix so it's passed to eBayes
contrast_matrix <- makeContrasts(Late_vs_Early = Late - Early, Age = Age, levels = design)

fit <- lmFit(expr_aligned, design)
fit_contrast <- contrasts.fit(fit, contrast_matrix)
fit_bayes <- eBayes(fit_contrast)

# Get DGE results for the main contrast (Late vs Early)
dge_results <- topTable(fit_bayes, number = Inf, coef = "Late_vs_Early") %>%
  rownames_to_column("GeneSymbol")

# Get DGE results for the Age coefficient
dge_results_age <- topTable(fit_bayes, number = Inf, coef = "Age") %>%
  rownames_to_column("GeneSymbol")

# Merge results to get significance for both factors
dge_full_results <- left_join(
  dge_results,
  dge_results_age %>% dplyr::select(GeneSymbol, logFC_age = logFC, P.Value_age = P.Value, adj.P.Val_age = adj.P.Val),
  by = "GeneSymbol"
)

message("DGEA complete. Full results for all factors merged.")
write.csv(dge_full_results, "unified_analysis_outputs/DGE_Full_Results_Fibrosis_and_Age.csv", row.names = FALSE)
message("-> Full DGE results saved for reference.")

# --- 3.5. COMPLEX VOLCANO PLOT VISUALIZATION ---
message("\n--- 3.5. Generating Complex Volcano Plot ---")

P_ADJ_CUTOFF <- 0.05
LOG_FC_CUTOFF <- 0.5 # Using the 0.5 cutoff from the script

plot_data <- dge_full_results %>%
  mutate(
    sig_fibrosis = adj.P.Val < P_ADJ_CUTOFF & abs(logFC) > LOG_FC_CUTOFF,
    sig_age = adj.P.Val_age < P_ADJ_CUTOFF,

    significance_group = case_when(
      sig_fibrosis & sig_age ~ "Fibrosis & Age",
      sig_fibrosis & !sig_age ~ "Fibrosis Only",
      !sig_fibrosis & sig_age ~ "Age Only",
      TRUE ~ "Not Significant"
    ),

    # Set factor levels for correct plotting order
    significance_group = factor(significance_group, levels = c("Fibrosis & Age", "Fibrosis Only", "Age Only", "Not Significant"))
  )

# Add data for labeling top genes
plot_data <- plot_data %>%
  mutate(
    gene_label = ifelse(significance_group == "Fibrosis Only" | significance_group == "Fibrosis & Age", GeneSymbol, NA),
    # Rank by fibrosis p-value for labeling
    label_rank = dense_rank(adj.P.Val)
  )

# Define custom colors
sig_colors <- c(
  "Fibrosis & Age" = "purple",
  "Fibrosis Only" = "red",
  "Age Only" = "blue",
  "Not Significant" = "grey80"
)

volcano_plot <- ggplot(plot_data, aes(x = logFC, y = -log10(adj.P.Val))) +
  # Draw points
  geom_point(aes(color = significance_group, alpha = significance_group), size = 1.5) +

  # Add labels for top 20 genes (ranked by fibrosis p-val)
  geom_text_repel(
    data = subset(plot_data, label_rank <= 20),
    aes(label = gene_label),
    size = 3,
    max.overlaps = 20,
    box.padding = 0.5,
    point.padding = 0.2
  ) +

  # Set colors and alpha
  scale_color_manual(values = sig_colors) +
  scale_alpha_manual(values = c("Fibrosis & Age" = 1, "Fibrosis Only" = 1, "Age Only" = 0.8, "Not Significant" = 0.5)) +

  # Add cutoff lines
  geom_hline(yintercept = -log10(P_ADJ_CUTOFF), linetype = "dashed", color = "grey50") +
  geom_vline(xintercept = LOG_FC_CUTOFF, linetype = "dashed", color = "grey50") +
  geom_vline(xintercept = -LOG_FC_CUTOFF, linetype = "dashed", color = "grey50") +

  # Labels and theme
  labs(
    title = "Volcano Plot: Late vs Early Fibrosis (Corrected for Age)",
    x = "log2(Fold Change)",
    y = "-log10(Adjusted P-Value)",
    color = "Significance Group"
  ) +
  theme_bw(base_size = 12) +
  theme(
    legend.position = "bottom",
    legend.title = element_text(face = "bold")
  ) +
  guides(alpha = "none") # Hide alpha from legend

ggsave("unified_analysis_outputs/figures/VolcanoPlot_Fibrosis_vs_Age.png", volcano_plot, width = 10, height = 8)
message("-> Complex volcano plot saved.")


# --- 4. PREPARE GENE LISTS FOR ENRICHMENT ---
message("\n--- 4. Preparing Gene Lists for Downstream Analysis ---")

significant_genes_df <- dge_full_results %>%
  filter(adj.P.Val < P_ADJ_CUTOFF & abs(logFC) > LOG_FC_CUTOFF)

message(paste("Found", nrow(significant_genes_df), "significant DEGs for Fibrosis contrast."))

entrez_map <- bitr(significant_genes_df$GeneSymbol, fromType="SYMBOL", toType="ENTREZID", OrgDb="org.Hs.eg.db")
significant_entrez <- na.omit(entrez_map$ENTREZID)

background_map <- bitr(dge_full_results$GeneSymbol, fromType="SYMBOL", toType="ENTREZID", OrgDb="org.Hs.eg.db")
background_entrez <- na.omit(background_map$ENTREZID)
message(paste("Prepared", length(significant_entrez), "significant genes with Entrez IDs for enrichment."))

# --- 5. MULTI-DATABASE PATHWAY ENRICHMENT ANALYSIS ---
message("\n--- 5. Running Enrichment Analysis (GO, KEGG, Reactome, MSigDB) ---")

# Run enrichments and save results
try({
  go_enrich <- enrichGO(gene=significant_entrez, universe=background_entrez, OrgDb=org.Hs.eg.db, ont="BP", pAdjustMethod="BH", qvalueCutoff=0.05, readable=TRUE)
  if(nrow(go_enrich) > 0) write.csv(as.data.frame(go_enrich), "unified_analysis_outputs/enrichment_results/GO_BP_enrichment.csv")
  message(paste("GO BP analysis found", nrow(go_enrich), "terms."))
}, silent = TRUE)

try({
  kegg_enrich <- enrichKEGG(gene=significant_entrez, universe=background_entrez, organism='hsa', pAdjustMethod="BH", qvalueCutoff=0.05)
  if(nrow(kegg_enrich) > 0) {
    kegg_enrich <- setReadable(kegg_enrich, OrgDb = org.Hs.eg.db, keyType = "ENTREZID")
    write.csv(as.data.frame(kegg_enrich), "unified_analysis_outputs/enrichment_results/KEGG_enrichment.csv")
    message(paste("KEGG analysis found", nrow(kegg_enrich), "terms."))
  }
}, silent = TRUE)

try({
  reactome_enrich <- enrichPathway(gene=significant_entrez, universe=background_entrez, organism="human", pAdjustMethod="BH", qvalueCutoff=0.05)
  if(nrow(reactome_enrich) > 0) {
    reactome_enrich <- setReadable(reactome_enrich, OrgDb = org.Hs.eg.db, keyType = "ENTREZID")
    write.csv(as.data.frame(reactome_enrich), "unified_analysis_outputs/enrichment_results/Reactome_enrichment.csv")
    message(paste("Reactome analysis found", nrow(reactome_enrich), "terms."))
  }
}, silent = TRUE)

try({
  msigdb_h_df <- msigdbr(species = "Homo sapiens", collection = "H")

  # Check for the correct Entrez ID column
  if ("ncbi_gene" %in% colnames(msigdb_h_df)) {
    entrez_col_name <- "ncbi_gene"
  } else if ("entrez_gene" %in% colnames(msigdb_h_df)) {
    entrez_col_name <- "entrez_gene"
  } else {
    stop("FATAL ERROR: Neither 'ncbi_gene' nor 'entrez_gene' column found in msigdbr output.")
  }

  msigdb_hallmark <- msigdb_h_df %>%
    dplyr::select(gs_name, !!sym(entrez_col_name)) %>%
    dplyr::rename(entrez_gene = !!sym(entrez_col_name))

  msigdb_enrich <- enricher(gene=significant_entrez, universe=background_entrez, TERM2GENE=msigdb_hallmark, pAdjustMethod="BH", qvalueCutoff=0.05)
  if(nrow(msigdb_enrich) > 0) {
    msigdb_enrich <- setReadable(msigdb_enrich, OrgDb = org.Hs.eg.db, keyType = "ENTREZID")
    write.csv(as.data.frame(msigdb_enrich), "unified_analysis_outputs/enrichment_results/MSigDB_Hallmark_enrichment.csv")
    message(paste("MSigDB Hallmark analysis found", nrow(msigdb_enrich), "terms."))
  }
}, silent = TRUE)

message("Enrichment analyses complete and results saved.")


# --- 6. VISUALIZATION OF ENRICHMENT RESULTS ---
message("\n--- 6. Generating Enrichment Visualizations ---")

# GO Plots
if (exists("go_enrich") && !is.null(go_enrich) && nrow(go_enrich) > 0) {
  dot_go <- dotplot(go_enrich, showCategory=20) + ggtitle("Top 20 Enriched GO Biological Processes")
  ggsave("unified_analysis_outputs/figures/BubblePlot_GO_Enrichment.png", dot_go, width=10, height=8)

  go_enrich_pairsim <- pairwise_termsim(go_enrich)
  emap_go <- emapplot(go_enrich_pairsim, showCategory=30) + ggtitle("GO Enrichment Map")
  ggsave("unified_analysis_outputs/figures/EnrichmentMap_GO.png", emap_go, width=12, height=10)
  message("-> GO Bubble Plot and Enrichment Map saved.")

  # --- GOCircle Plot (Replaces GOChord) ---

  # 1. Prepare gene data
  gene_data_for_plot <- dge_results %>%
    dplyr::select(GeneSymbol, logFC) %>%
    distinct(GeneSymbol, .keep_all = TRUE) # Ensure unique gene symbols

  # 2. Prepare GO term data
  go_results_df <- as.data.frame(go_enrich) %>%
    dplyr::select(ID, Description, p.adjust, geneID)

  if (nrow(go_results_df) >= 5) { # GOCircle looks best with 5-10 terms
    # 3. Select top 10 terms
    top_10_terms <- go_results_df[1:10, ]

    # --- ROBUST FIX for circle_dat ---
    # 4. Get the list of all valid genes we have logFC data for
    valid_genes <- gene_data_for_plot$GeneSymbol

    # 5. Clean the geneID list in each term
    cleaned_geneID_list <- lapply(top_10_terms$geneID, function(gene_string) {
      genes_in_term <- unlist(strsplit(gene_string, "/"))
      genes_to_keep <- genes_in_term[genes_in_term %in% valid_genes] # Find intersection
      return(paste(genes_to_keep, collapse = "/"))
    })

    top_10_terms$geneID <- unlist(cleaned_geneID_list)

    # 6. Filter out terms that now have no matching genes
    top_10_terms_clean <- top_10_terms %>%
      filter(geneID != "")

    # 7. Check if any terms survived
    if (nrow(top_10_terms_clean) > 0) {

      # 8. Use GOplot::circle_dat to format data
      circ_data <- try(circle_dat(terms = top_10_terms_clean, genes = gene_data_for_plot))

      if (!inherits(circ_data, "try-error")) {
        png("unified_analysis_outputs/figures/CircosPlot_GO.png", width=12, height=12, units="in", res=300)
        GOCircle(circ_data,
                 nsub = nrow(top_10_terms_clean), # Show all surviving terms
                 lfc.col = c('blue', 'white', 'red'), # logFC heatmap colors
                 zsc.col = c('grey', 'black'), # z-score colors
                 label.size = 5,
                 term.col = "auto")
        dev.off()
        message("-> GOCircle plot saved.")
      } else {
        message(paste("Skipping GOCircle plot: Error in circle_dat function even after cleaning:", circ_data[1]))
      }
    } else {
      message("Skipping GOCircle plot: No top GO terms had genes matching the DGE results.")
    }

  } else {
    message("Not enough significant GO terms (need at least 5) to generate a GOCircle plot.")
  }

} else {
  message("No significant GO terms found; skipping GO visualizations.")
}

# Other Enrichment Plots
if (exists("kegg_enrich") && !is.null(kegg_enrich) && nrow(kegg_enrich) > 0) {
  dot_kegg <- dotplot(kegg_enrich, showCategory=20) + ggtitle("Top 20 Enriched KEGG Pathways")
  ggsave("unified_analysis_outputs/figures/BubblePlot_KEGG.png", dot_kegg, width=10, height=8)
  message("-> KEGG Bubble Plot saved.")

  kegg_pairsim <- pairwise_termsim(kegg_enrich)
  emap_kegg <- emapplot(kegg_pairsim, showCategory = 30) + ggtitle("KEGG Enrichment Map")
  ggsave("unified_analysis_outputs/figures/EnrichmentMap_KEGG.png", emap_kegg, width = 12, height = 10)
  message("-> KEGG Enrichment Map saved.")
}

if (exists("reactome_enrich") && !is.null(reactome_enrich) && nrow(reactome_enrich) > 0) {
  dot_reactome <- dotplot(reactome_enrich, showCategory=20) + ggtitle("Top 20 Enriched Reactome Pathways")
  ggsave("unified_analysis_outputs/figures/BubblePlot_Reactome.png", dot_reactome, width=10, height=8)
  message("-> Reactome Bubble Plot saved.")

  reactome_pairsim <- pairwise_termsim(reactome_enrich)
  emap_reactome <- emapplot(reactome_pairsim, showCategory = 30) + ggtitle("Reactome Enrichment Map")
  ggsave("unified_analysis_outputs/figures/EnrichmentMap_Reactome.png", emap_reactome, width = 12, height = 10)
  message("-> Reactome Enrichment Map saved.")
}

if (exists("msigdb_enrich") && !is.null(msigdb_enrich) && nrow(msigdb_enrich) > 0) {
  dot_msigdb <- dotplot(msigdb_enrich, showCategory=20) + ggtitle("Top 20 Enriched MSigDB Hallmarks")
  ggsave("unified_analysis_outputs/figures/BubblePlot_MSigDB.png", dot_msigdb, width=10, height=8)
  message("-> MSigDB Hallmark Bubble Plot saved.")

  msigdb_pairsim <- pairwise_termsim(msigdb_enrich)
  emap_msigdb <- emapplot(msigdb_pairsim, showCategory = 30) + ggtitle("MSigDB Hallmark Enrichment Map")
  ggsave("unified_analysis_outputs/figures/EnrichmentMap_MSigDB.png", emap_msigdb, width = 12, height = 10)
  message("-> MSigDB Hallmark Enrichment Map saved.")
}


# --- 7. PROTEIN-PROTEIN INTERACTION (PPI) NETWORK CONSTRUCTION ---
message("\n--- 7. Building PPI Network from STRING ---")

if (nrow(significant_genes_df) > 0) {
  string_db <- STRINGdb$new(version="11.5", species=9606, score_threshold=400, input_directory="")

  # Map DEGs to STRING IDs and include their logFC values
  string_map <- string_db$map(significant_genes_df, "GeneSymbol", removeUnmappedRows = TRUE)

  if (nrow(string_map) > 0) {
    # Get interactions
    string_interactions <- string_db$get_interactions(string_map$STRING_id) # Uses STRING_id

    # Prepare links and nodes for igraph
    # FIX: The interaction score column from STRINGdb is 'combined_score', not 'score'
    links <- string_interactions %>%
      dplyr::select(from, to, combined_score) %>%
      dplyr::rename(weight = combined_score)

    # --- FIX: Use 'STRING_id' (lowercase) not 'STRING_ID' ---
    nodes <- string_map %>%
      dplyr::select(STRING_id, GeneSymbol, logFC, adj.P.Val) %>%
      dplyr::rename(name = STRING_id) %>%
      distinct(name, .keep_all = TRUE) # Ensure nodes are unique

    # Create igraph object
    net <- graph_from_data_frame(d=links, vertices=nodes, directed=FALSE)

    # Simplify network (remove loops and multiple edges)
    net <- simplify(net, remove.multiple = TRUE, remove.loops = TRUE)

    # --- 8. PPI NETWORK VISUALIZATION & EXPORT ---
    message("\n--- 8. Visualizing and Exporting PPI Network ---")

    # Get the largest connected component
    components <- clusters(net, mode="weak")
    main_component <- induced_subgraph(net, V(net)[components$membership == which.max(components$csize)])
    message(paste("PPI network built. Main component has", vcount(main_component), "nodes and", ecount(main_component), "edges."))

    # Set visual attributes
    V(main_component)$size <- (degree(main_component) / max(degree(main_component)) * 10) + 3 # Scale size

    # Create color palette (blue -> white -> red)
    logFC_values <- V(main_component)$logFC
    max_abs_logFC <- max(abs(logFC_values), na.rm = TRUE)
    if(is.infinite(max_abs_logFC)) max_abs_logFC <- max(abs(logFC_values[is.finite(logFC_values)]), na.rm = TRUE) # Handle Inf

    color_breaks <- seq(-max_abs_logFC, max_abs_logFC, length.out = 100)
    pal <- colorRampPalette(c("blue", "white", "red"))(99)

    # Assign colors based on logFC
    V(main_component)$color <- pal[cut(logFC_values, breaks = color_breaks, include.lowest = TRUE)]
    V(main_component)$color[is.na(V(main_component)$color)] <- "grey" # For any NA logFCs

    V(main_component)$label <- V(main_component)$GeneSymbol
    V(main_component)$label.cex <- 0.7
    V(main_component)$label.color <- "black"

    # Save network plot
    png("unified_analysis_outputs/figures/ForceDirected_PPI_Network.png", width=12, height=12, units="in", res=300)
    plot(main_component,
         layout=layout_with_fr,
         vertex.frame.color="black",
         main="PPI Network of DEGs (Main Component)")
    # Add a simple legend
    legend("bottomright", legend=c("Upregulated", "Downregulated"), fill=c("red", "blue"), bty="n", title="logFC")
    dev.off()
    message("-> PPI Network plot saved.")

    # Export network files
    write_graph(main_component, "unified_analysis_outputs/networks/PPI_Network.graphml", format = "graphml")

    edge_list <- as_data_frame(main_component, what = "edges")
    node_list <- as_data_frame(main_component, what = "vertices")

    # Map STRING IDs back to GeneSymbols in edge list for readability
    edge_list_symbols <- edge_list %>%
      left_join(node_list %>% dplyr::select(name, GeneSymbol), by = c("from" = "name")) %>%
      dplyr::rename(from_gene = GeneSymbol) %>%
      left_join(node_list %>% dplyr::select(name, GeneSymbol), by = c("to" = "name")) %>%
      dplyr::rename(to_gene = GeneSymbol) %>%
      dplyr::select(from_gene, to_gene, weight)

    write.csv(edge_list_symbols, "unified_analysis_outputs/networks/PPI_Network_EdgeList.csv", row.names = FALSE)
    write.csv(node_list, "unified_analysis_outputs/networks/PPI_Network_NodeList.csv", row.names = FALSE)
    message("-> Network .graphml and .csv files saved.")

    # --- NEW: Export .sif file ---
    sif_data <- edge_list_symbols %>%
      mutate(interaction_type = "pp") %>% # 'pp' for protein-protein interaction
      dplyr::select(from_gene, interaction_type, to_gene)

    write_tsv(sif_data, "unified_analysis_outputs/networks/PPI_Network.sif", col_names = FALSE)
    message("-> Network .sif file saved.")

  } else {
    message("No significant DEGs could be mapped to STRING. Skipping PPI network.")
  }
} else {
  message("No significant DEGs found. Skipping PPI network construction.")
}

# --- 9. DGE HEATMAP VISUALIZATION ---
message("\n--- 9. Generating DGE Heatmap ---")

# --- NEW: Load ferroptosis gene lists ---
if (file.exists("ferroptosis_driver.csv") && file.exists("ferroptosis_suppressor.csv")) {
  ferro_drivers <- read.csv("ferroptosis_driver.csv")
  ferro_suppressors <- read.csv("ferroptosis_suppressor.csv")

  driver_genes <- unique(ferro_drivers$symbol)
  suppressor_genes <- unique(ferro_suppressors$symbol)
  message(paste("Loaded", length(driver_genes), "ferroptosis drivers and", length(suppressor_genes), "suppressors."))

  ferroptosis_annotation_available <- TRUE
} else {
  message("Ferroptosis files not found. Skipping heatmap annotation.")
  ferroptosis_annotation_available <- FALSE
}


if (nrow(significant_genes_df) > 0) {
  # Get expression data for significant genes
  sig_gene_symbols <- significant_genes_df$GeneSymbol

  # Ensure genes exist in the expression matrix
  genes_present <- sig_gene_symbols[sig_gene_symbols %in% rownames(expr_aligned)]
  expr_sig <- expr_aligned[genes_present, ]

  if (nrow(expr_sig) > 1) { # pheatmap needs at least 2 rows
    # Prepare column annotation
    annotation_col <- meta_clean[, "fibrosis_group", drop = FALSE]

    # --- NEW: Prepare row annotation for ferroptosis ---

    # Default pheatmap parameters
    pheatmap_args <- list(
      mat = expr_sig,
      scale = "row",
      annotation_col = annotation_col,
      show_rownames = TRUE, # Show gene symbols
      fontsize_row = 12,    # Set font size to 12 as requested
      show_colnames = FALSE,
      cluster_cols = TRUE,
      cluster_rows = TRUE,
      main = "Heatmap of Differentially Expressed Genes",
      color = colorRampPalette(c("blue", "white", "red"))(100)
    )

    # --- FIX for annotation_colors ---
    # Define the colors for the column annotation
    col_colors <- list(
      fibrosis_group = c("Early" = "lightblue", "Late" = "darkblue")
    )

    if (ferroptosis_annotation_available) {
      # Create the annotation data frame
      gene_annotation_df <- data.frame(
        GeneSymbol = rownames(expr_sig)
      ) %>%
        mutate(
          Ferroptosis = case_when(
            GeneSymbol %in% driver_genes ~ "Driver",
            GeneSymbol %in% suppressor_genes ~ "Suppressor",
            TRUE ~ "Not Annotated"
          )
        ) %>%
        column_to_rownames("GeneSymbol")

      # Define the colors for the row annotation
      row_colors <- list(
        Ferroptosis = c(
          "Driver" = "red",
          "Suppressor" = "green",
          "Not Annotated" = "grey90"
        )
      )

      # Combine annotation color lists
      pheatmap_args$annotation_colors <- c(col_colors, row_colors)

      # Add row annotation
      pheatmap_args$annotation_row <- gene_annotation_df

      # Filter out the "Not Annotated" to make the plot cleaner
      # We only keep rows that ARE annotated
      genes_to_show <- rownames(gene_annotation_df)[gene_annotation_df$Ferroptosis != "Not Annotated"]

      if(length(genes_to_show) > 1) {
        # If we have annotated genes, filter the matrix and annotation
        pheatmap_args$mat <- expr_sig[genes_to_show, ]
        pheatmap_args$annotation_row <- gene_annotation_df[genes_to_show, , drop = FALSE]
        pheatmap_args$main <- "Heatmap of DEGs related to Ferroptosis"
        # Adjust font size if too many genes
        if(length(genes_to_show) > 50) pheatmap_args$fontsize_row <- 8
      } else {
        message("No significant DEGs found in ferroptosis lists. Showing all DEGs without annotation.")
        # Remove the row annotation args if no genes match
        pheatmap_args$annotation_row <- NULL
        pheatmap_args$annotation_colors <- col_colors # Only use col_colors
        pheatmap_args$show_rownames <- FALSE # Too many genes
        pheatmap_args$fontsize_row <- 8
      }

    } else {
      pheatmap_args$show_rownames <- FALSE # Too many genes if not filtering
      pheatmap_args$fontsize_row <- 8
      pheatmap_args$annotation_colors <- col_colors # Only use col_colors
    }

    # Create plot
    png("unified_analysis_outputs/figures/Heatmap_DEGs.png", width=10, height=12, units="in", res=300)
    do.call(pheatmap, pheatmap_args)
    dev.off()
    message("-> DGE Heatmap saved.")

  } else {
    message("Skipping heatmap: Not enough significant genes (need > 1) to cluster.")
  }
} else {
  message("Skipping heatmap: No significant genes found.")
}

message("\n--- UNIFIED PIPELINE COMPLETE ---")

# --- Refined Ferroptosis Heatmap Generation ---
message("\n--- Generating Specialized Ferroptosis DEG Heatmap ---")

# 1. Prepare Column Annotation (Clinical Metadata)
# We select Age, Fibrosis Group, and the raw Fibrosis Stage
annotation_col <- meta_clean %>%
  dplyr::select(age, fibrosis_group, fibrosis_stage)

# 2. Identify Ferroptosis DEGs
# Filter the expression matrix for genes that are both DEGs and in our Ferroptosis lists
ferro_genes_all <- c(driver_genes, suppressor_genes)
sig_ferro_genes <- intersect(significant_genes_df$GeneSymbol, ferro_genes_all)

if (length(sig_ferro_genes) > 1) {

  # Filter expression matrix
  expr_ferro <- expr_aligned[sig_ferro_genes, ]

  # 3. Prepare Row Annotation (Gene Function)
  gene_annotation_row <- data.frame(
    GeneSymbol = sig_ferro_genes
  ) %>%
    mutate(
      Role = case_when(
        GeneSymbol %in% driver_genes ~ "Driver",
        GeneSymbol %in% suppressor_genes ~ "Suppressor",
        TRUE ~ "Unknown"
      )
    ) %>%
    column_to_rownames("GeneSymbol")

  # 4. Define Aesthetic Colors
  ann_colors <- list(
    fibrosis_group = c("Early" = "#A6CEE3", "Late" = "#1F78B4"),
    fibrosis_stage = c("0"="#F7FCF0", "1"="#E0F3DB", "2"="#A8DDB5", "3"="#4EB3D3", "4"="#084081"),
    Role = c("Driver" = "#E31A1C", "Suppressor" = "#33A02C"),
    # For continuous 'age', pheatmap will use a default gradient unless specified
    age = colorRampPalette(c("white", "orange"))(100)
  )

  # 5. Generate and Save the Heatmap
  png("unified_analysis_outputs/figures/Heatmap_Ferroptosis_Detailed.png",
      width=12, height=10, units="in", res=300)

  pheatmap(
    mat = expr_ferro,
    scale = "row",
    clustering_distance_rows = "correlation",
    clustering_distance_cols = "euclidean",
    clustering_method = "complete",
    annotation_col = annotation_col,
    annotation_row = gene_annotation_row,
    annotation_colors = ann_colors,
    show_colnames = FALSE,
    show_rownames = TRUE,
    fontsize_row = 10,
    main = "Ferroptosis DEGs: Expression by Fibrosis Severity and Age",
    color = colorRampPalette(c("blue", "white", "red"))(100)
  )

  dev.off()
  message("-> Specialized Ferroptosis heatmap saved to figures directory.")

} else {
  message("Insufficient Ferroptosis DEGs found to generate a heatmap.")
}
