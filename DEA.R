library(limma)
library(ggplot2)
library(dplyr)
library(ggrepel)
library(patchwork) 

base_dir <- dirname(rstudioapi::getActiveDocumentContext()$path)
setwd(base_dir)

perform_limma_dea <- function(cancer_label, source_name,
                              expr_thresh = 1,
                              min_prop = 0.2,
                              fc_thresh = 1.0,
                              fdr_thresh = 0.05) {
  
  cat("=====================================\n")
  cat("Processing:", cancer_label, "vs Normal\n")
  cat("=====================================\n")
  
  # ── Load data ──
  filename <- file.path("outputs", "DEA", paste0(cancer_label, "_ge.csv"))
  bn_df <- read.csv(filename, row.names = 1, check.names = FALSE)
  expr_matrix <- as.matrix(bn_df)
  cat("Loaded:", nrow(expr_matrix), "genes ×", ncol(expr_matrix), "samples\n")
  
  # ── Verify data is log2-transformed ──
  cat("\nData range check (should be ~0-25 for log2 data):\n")
  cat("  Min:", round(min(expr_matrix), 2), 
      " Median:", round(median(expr_matrix), 2),
      " Max:", round(max(expr_matrix), 2), "\n")
  
  sample_names <- colnames(expr_matrix)
  group <- factor(
    ifelse(grepl("^GTEX", sample_names), "Normal", cancer_label),
    levels = c("Normal", cancer_label)
  )
  cat(cancer_label, "samples:", sum(group == cancer_label),
      "| Normal samples:", sum(group == "Normal"), "\n")
  
  cancer_idx <- which(group == cancer_label)
  normal_idx <- which(group == "Normal")
  
  prop_cancer <- rowMeans(expr_matrix[, cancer_idx] > expr_thresh)
  prop_normal <- rowMeans(expr_matrix[, normal_idx] > expr_thresh)
  keep <- (prop_cancer >= min_prop) | (prop_normal >= min_prop)
  
  cat("\n=== FILTERING LOW EXPRESSED GENES ===\n")
  cat("  Criterion: log2(count+1) >", expr_thresh, 
      "in ≥", min_prop * 100, "% of samples in at least one group\n")
  cat("  Retained:", sum(keep), "/", nrow(expr_matrix), "genes\n\n")
  
  expr_filtered <- expr_matrix[keep, ]

  cat("=== PCA BATCH EFFECT CHECK & OUTLIER REMOVAL ===\n")
  
  # Initial PCA to find outliers
  pca_init <- prcomp(t(expr_filtered), scale. = TRUE)
  pc1_scores <- pca_init$x[, 1]
  pc2_scores <- pca_init$x[, 2]
  
  # Define Outliers: More than 3 Standard Deviations from the mean on PC1 or PC2
  z_pc1 <- abs((pc1_scores - mean(pc1_scores)) / sd(pc1_scores))
  z_pc2 <- abs((pc2_scores - mean(pc2_scores)) / sd(pc2_scores))
  outliers <- which(z_pc1 > 3 | z_pc2 > 3)
  
  if (length(outliers) > 0) {
    cat("  Found", length(outliers), "extreme outliers. Removing samples:", 
        paste(colnames(expr_filtered)[outliers], collapse=", "), "\n")
    
    # Remove outliers from expression matrix and group factor
    expr_filtered <- expr_filtered[, -outliers]
    group <- group[-outliers]
    
    pca <- prcomp(t(expr_filtered), scale. = TRUE)
  } else {
    cat("  No extreme outliers detected.\n")
    pca <- pca_init
  }
  
  var_explained <- summary(pca)$importance[2, 1:2] * 100
  pca_df <- data.frame(
    PC1 = pca$x[, 1],
    PC2 = pca$x[, 2],
    Group = group,
    Source = ifelse(grepl("^GTEX", colnames(expr_filtered)), "GTEx", "TCGA")
  )
  pca_plot <- ggplot(pca_df, aes(x = PC1, y = PC2, color = Group, shape = Source)) +
    geom_point(alpha = 0.6, size = 2) +
    scale_color_manual(values = c("Normal" = "#00AFBB", setNames("#bb0c00", cancer_label))) +
    labs(
      title = paste0("PCA: ", cancer_label, " vs Normal — Batch Effect Check"),
      x = paste0("PC1 (", round(var_explained[1], 1), "% variance)"),
      y = paste0("PC2 (", round(var_explained[2], 1), "% variance)")
    ) +
    theme_minimal() +
    theme(
      plot.title = element_text(size = 14, face = "bold"),
      legend.position = "bottom"
    )
  
  # Save PCA plot
  pca_output_dir <- file.path("outputs", "DEA_results", paste0(cancer_label, "_", source_name))
  if (!dir.exists(pca_output_dir)) dir.create(pca_output_dir, recursive = TRUE)
  ggsave(file.path(pca_output_dir, "pca_batch_check.png"),
         pca_plot, width = 10, height = 8, dpi = 300)
  cat("  PCA plot saved.\n")
  cat("  PC1 explains:", round(var_explained[1], 1), "% variance\n")
  cat("  PC2 explains:", round(var_explained[2], 1), "% variance\n\n")
  
  design <- model.matrix(~ group)
  colnames(design) <- c("Intercept", paste0(cancer_label, "_vs_Normal"))
  
  cat("Applying limma on log2(expected_count+1) data...\n")
  
  fit <- lmFit(expr_filtered, design)
  fit <- eBayes(fit, trend = TRUE)
  
  dea_results <- topTable(fit, coef = 2, number = Inf, sort.by = "P")
  dea_results$Gene <- rownames(dea_results)
  dea_results <- dea_results %>%
    dplyr::select(Gene, logFC, AveExpr, PValue = P.Value, adj.P.Val, t, B)
  
  # ── Summary ──
  cat("\nDEA Summary:\n")
  cat("Total genes tested:", nrow(dea_results), "\n")
  cat("Significant (FDR <", fdr_thresh, "):", 
      sum(dea_results$adj.P.Val < fdr_thresh, na.rm = TRUE), "\n")
  cat("Upregulated (FDR<", fdr_thresh, "& logFC>", fc_thresh, "):", 
      sum(dea_results$adj.P.Val < fdr_thresh & dea_results$logFC > fc_thresh, na.rm = TRUE), "\n")
  cat("Downregulated (FDR<", fdr_thresh, "& logFC<", -fc_thresh, "):", 
      sum(dea_results$adj.P.Val < fdr_thresh & dea_results$logFC < -fc_thresh, na.rm = TRUE), "\n\n")
  
  # ── Save results ──
  output_dir <- file.path("outputs", "DEA_results", paste0(cancer_label, "_", source_name))
  write.csv(dea_results, file.path(output_dir, "full_dea_results.csv"), row.names = FALSE)
  
  # ── Top genes & volcano plot ──
  top_genes_file <- file.path("outputs", "DEA", 
                              paste0("top_", cancer_label, "_", source_name, "_genes.csv"))
  top_genes_df <- read.csv(top_genes_file, header = FALSE)
  top_genes_list <- trimws(unlist(strsplit(as.character(top_genes_df[1, ]), ",")))
  
  volcano_plot <- create_volcano_plot(dea_results, cancer_label, source_name, 
                                      top_genes_list, fc_thresh, fdr_thresh)
  ggsave(file.path(output_dir, "volcano_plot_with_top_genes.png"),
         volcano_plot, width = 12, height = 10, dpi = 300)
  
  analyze_top_genes_regulation(dea_results, cancer_label, top_genes_list, 
                               output_dir, fc_thresh, fdr_thresh)
  
  return(volcano_plot)
}


create_volcano_plot <- function(results, cancer_label, source_name, 
                                top_genes_list, fc_thresh = 1.0, fdr_thresh = 0.05) {
  
  plot_data <- results %>%
    mutate(
      log10FDR = -log10(adj.P.Val),
      Significance = case_when(
        adj.P.Val < fdr_thresh & logFC > fc_thresh ~ "Upregulated",
        adj.P.Val < fdr_thresh & logFC < -fc_thresh ~ "Downregulated",
        TRUE ~ "Not significant"
      )
    )
  
  x_limit <- quantile(abs(plot_data$logFC), 0.995, na.rm = TRUE) * 1.1
  cat("  Volcano x-axis limit: ±", round(x_limit, 2), "\n")
  cat("  Genes beyond this limit:", 
      sum(abs(plot_data$logFC) > x_limit), "/", nrow(plot_data), "\n")
  
  n_up <- sum(plot_data$Significance == "Upregulated", na.rm = TRUE)
  n_down <- sum(plot_data$Significance == "Downregulated", na.rm = TRUE)
  n_ns <- sum(plot_data$Significance == "Not significant", na.rm = TRUE)
  
  # ── Only biologically relevant top genes (FDR < thresh AND |logFC| > thresh) ──
  top_genes_data <- plot_data %>%
    filter(Gene %in% top_genes_list & 
             adj.P.Val < fdr_thresh & 
             abs(logFC) > fc_thresh)
  
  p <- ggplot(plot_data, aes(x = logFC, y = log10FDR, color = Significance)) +
    geom_point(alpha = 0.4, size = 1.2, shape = 19, stroke = 0) +
    scale_color_manual(values = c(
      "Downregulated" = "#00AFBB",
      "Upregulated" = "#bb0c00",
      "Not significant" = "#CCCCCC"
    )) +
    guides(color = guide_legend(override.aes = list(size = 3))) +
    geom_hline(yintercept = -log10(fdr_thresh), linetype = "dashed", 
               color = "darkgray", linewidth = 0.5) +
    geom_vline(xintercept = c(-fc_thresh, fc_thresh), linetype = "dashed", 
               color = "darkgray", linewidth = 0.5) +
    coord_cartesian(xlim = c(-x_limit, x_limit)) +
    labs(x = "Expression Change (log2 FC)",
         y = "Statistical Significance (-log10 FDR)") +
    theme_minimal() +
    theme(
      axis.title = element_text(size = 16),
      legend.text = element_text(size = 12),
      legend.position = "bottom",
      plot.subtitle = element_text(size = 11),
      panel.grid = element_blank(),
      legend.title = element_blank()
    )
  
  # ── Add only biologically relevant top genes ──
  if (nrow(top_genes_data) > 0) {
    p <- p +
      geom_point(data = top_genes_data, color = "black", size = 2, shape = 19) +
      geom_text_repel(data = top_genes_data, aes(label = Gene),
                      color = "black", size = 3.5, fontface = "bold",
                      max.overlaps = 30, box.padding = 0.3,
                      segment.color = "black", segment.alpha = 0.4,
                      show.legend = FALSE)
  }
  
  return(p)
}


analyze_top_genes_regulation <- function(results, cancer_label, top_genes_list, 
                                         output_dir, fc_thresh = 1.0, fdr_thresh = 0.05) {
  
  matches <- top_genes_list %in% results$Gene
  cat("Exact matches found:", sum(matches), "/", length(top_genes_list), "\n")
  
  top_genes_results <- results[results$Gene %in% top_genes_list, ]
  if (nrow(top_genes_results) == 0) {
    cat("No top genes found in DE results\n")
    return(NULL)
  }
  
  cat("\n", rep("=", 70), "\n", sep = "")
  cat("TOP GENES REGULATION ANALYSIS:", cancer_label, "\n")
  cat(rep("=", 70), "\n", sep = "")
  
  top_genes_results$Regulation <- ifelse(
    top_genes_results$logFC > fc_thresh, "UPREGULATED",
    ifelse(top_genes_results$logFC < -fc_thresh, "DOWNREGULATED", "NO SIGNIFICANT CHANGE")
  )
  top_genes_results$Significance <- ifelse(
    top_genes_results$adj.P.Val < fdr_thresh, "SIGNIFICANT", "NOT SIGNIFICANT"
  )
  top_genes_results$Biologically_Relevant <- ifelse(
    top_genes_results$adj.P.Val < fdr_thresh & abs(top_genes_results$logFC) > fc_thresh,
    "YES", "NO"
  )
  
  top_genes_results <- top_genes_results[order(-abs(top_genes_results$logFC)), ]
  print(top_genes_results[, c("Gene", "logFC", "AveExpr", "adj.P.Val",
                              "Regulation", "Significance", "Biologically_Relevant")])
  
  cat("\nSUMMARY:\n")
  cat("Found:", nrow(top_genes_results), "/", length(top_genes_list), "\n")
  cat("Upregulated:", sum(top_genes_results$logFC > fc_thresh), "\n")
  cat("Downregulated:", sum(top_genes_results$logFC < -fc_thresh), "\n")
  cat("Significant:", sum(top_genes_results$adj.P.Val < fdr_thresh), "\n")
  cat("Biologically relevant:", sum(top_genes_results$adj.P.Val < fdr_thresh &
                                      abs(top_genes_results$logFC) > fc_thresh), "\n")
  
  missing <- setdiff(top_genes_list, top_genes_results$Gene)
  if (length(missing) > 0) {
    cat("\nMISSING GENES:\n")
    print(missing)
  }
  
  write.csv(top_genes_results, file.path(output_dir, "top_genes_analysis.csv"), row.names = FALSE)
  return(top_genes_results)
}


gbm_ge_volcano_plot <- perform_limma_dea('GBM', 'ge')
gbm_methyl_volcano_plot <- perform_limma_dea('GBM', 'methyl')

lgg_ge_volcano_plot <- perform_limma_dea('LGG', 'ge')
lgg_methyl_volcano_plot <- perform_limma_dea('LGG', 'methyl')

combined_gbm_plot <- (gbm_ge_volcano_plot + gbm_methyl_volcano_plot) +
  plot_layout(ncol = 2, guides = 'collect') +
  plot_annotation(tag_levels = 'A') &
  theme(
    plot.tag = element_text(size = 40, face = "bold"),
    legend.text = element_text(size = 16),
    legend.position = 'bottom'
  )

ggsave(file.path("outputs", "DEA_results", "GBM_combined_volcano.png"),
       combined_gbm_plot, width = 16, height = 8)

combined_lgg_plot <- (lgg_ge_volcano_plot + lgg_methyl_volcano_plot) +
  plot_layout(ncol = 2, guides = 'collect') +
  plot_annotation(tag_levels = 'A') &
  theme(
    plot.tag = element_text(size = 40, face = "bold"),
    legend.text = element_text(size = 16),
    legend.position = 'bottom'
  )

ggsave(file.path("outputs", "DEA_results", "LGG_combined_volcano.png"),
       combined_lgg_plot, width = 16, height = 8)



# ======================================================================================
# PART 2 - PCA ON DNA METHYLATION DATA ---> only run after having downloaded the data
# ======================================================================================

perform_methyl_pca <- function(cancer_label) {
  
  cat("=====================================\n")
  cat("Methylation PCA:", cancer_label, "vs Normal\n")
  cat("=====================================\n")
  
  filename <- file.path("outputs", "DEA", paste0(cancer_label, "_methyl.csv"))
  methyl_df <- read.csv(filename, row.names = 1, check.names = FALSE)
  methyl_matrix <- as.matrix(methyl_df)
  cat("Loaded:", nrow(methyl_matrix), "probes ×", ncol(methyl_matrix), "samples\n")
  
  sample_names <- colnames(methyl_matrix)
  group <- factor(
    ifelse(grepl("^GEO", sample_names), "Normal", cancer_label),
    levels = c("Normal", cancer_label)
  )
  cat(cancer_label, "samples:", sum(group == cancer_label),
      "| Normal samples:", sum(group == "Normal"), "\n")
  
  pca <- prcomp(t(methyl_matrix), scale. = TRUE)
  var_explained <- summary(pca)$importance[2, 1:2] * 100
  
  pca_df <- data.frame(
    PC1 = pca$x[, 1],
    PC2 = pca$x[, 2],
    Group = group,
    Source = ifelse(grepl("^GEO", colnames(methyl_matrix)), "GEO", "TCGA")
  )
  
  pca_plot <- ggplot(pca_df, aes(x = PC1, y = PC2, color = Group, shape = Source)) +
    geom_point(alpha = 0.6, size = 2) +
    scale_color_manual(values = c("Normal" = "#00AFBB", setNames("#bb0c00", cancer_label))) +
    labs(
      title = paste0("PCA: ", cancer_label, " vs Normal — DNA Methylation"),
      x = paste0("PC1 (", round(var_explained[1], 1), "% variance)"),
      y = paste0("PC2 (", round(var_explained[2], 1), "% variance)")
    ) +
    theme_minimal() +
    theme(
      plot.title = element_text(size = 14, face = "bold"),
      legend.position = "bottom"
    )
  
  output_dir <- file.path("outputs", "DEA_results", paste0(cancer_label, "_methyl"))
  if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)
  ggsave(file.path(output_dir, "pca_methylation.png"),
         pca_plot, width = 10, height = 8, dpi = 300)
  
  cat("  PCA plot saved.\n")
  cat("  PC1 explains:", round(var_explained[1], 1), "% variance\n")
  cat("  PC2 explains:", round(var_explained[2], 1), "% variance\n\n")
  
  return(pca_plot)
}

gbm_methyl_pca <- perform_methyl_pca("GBM")
lgg_methyl_pca <- perform_methyl_pca("LGG")
