library(limma)
library(ggplot2)
library(dplyr)
library(ggrepel)
library(patchwork)
base_dir <- dirname(rstudioapi::getActiveDocumentContext()$path)
setwd(base_dir)

perform_limma_dea <- function(cancer_label, source_name,
                              min_expr = 1.5,     
                              min_samples = 0.2) { 
  
  cat("=====================================\n")
  cat("Processing:", cancer_label, "vs Normal\n")
  cat("=====================================\n")
  
  filename <- file.path("outputs", "DEA", paste0(cancer_label, "_ge.csv"))
  bn_df <- read.csv(filename, row.names = 1, check.names = FALSE)
  
  # Transpose matrix
  expr_matrix <- t(as.matrix(bn_df[, colnames(bn_df) != cancer_label]))
  
  group <- factor(bn_df[[cancer_label]], levels = c(0, 1), 
                  labels = c("Normal", cancer_label))
  cat(cancer_label, "samples:", sum(group == cancer_label), 
      "| Normal samples:", sum(group == "Normal"), "\n")
  
  n_cancer <- sum(group == cancer_label)
  n_normal <- sum(group == "Normal")
  
  expr_cancer <- rowSums(expr_matrix[, group == cancer_label] > min_expr) >= ceiling(n_cancer * min_samples)
  expr_normal <- rowSums(expr_matrix[, group == "Normal"] > min_expr) >= ceiling(n_normal * min_samples)
  keep <- expr_cancer | expr_normal
  
  cat("\n=== FILTERING LOW EXPRESSED GENES ===\n")
  cat("  Keep genes if expressed in 20% of cancer OR normal samples\n")
  cat("  Retained:", sum(keep), "/", nrow(expr_matrix), "genes\n\n")
  
  expr_matrix <- expr_matrix[keep, ]
  
  design <- model.matrix(~ group)
  colnames(design) <- c("Intercept", paste0(cancer_label, "_vs_Normal"))
  
  cat("Fitting limma-trend model...\n")
  fit <- lmFit(expr_matrix, design)
  fit <- eBayes(fit, trend = TRUE)
  
  dea_results <- topTable(fit, coef = 2, number = Inf, sort.by = "P")
  dea_results$Gene <- rownames(dea_results)
  dea_results <- dea_results %>%
    dplyr::select(Gene, logFC, AveExpr, PValue = P.Value, adj.P.Val, t, B)
  
  cat("\nDEA Summary:\n")
  cat("Total genes tested:", nrow(dea_results), "\n")
  cat("Significant (FDR < 0.05):", sum(dea_results$adj.P.Val < 0.05, na.rm = TRUE), "\n")
  cat("Upregulated (FDR<0.05 & logFC>1.5):", sum(dea_results$adj.P.Val < 0.05 & dea_results$logFC > 1.5, na.rm = TRUE), "\n")
  cat("Downregulated (FDR<0.05 & logFC< -1):", sum(dea_results$adj.P.Val < 0.05 & dea_results$logFC < -1.5, na.rm = TRUE), "\n\n")
  
  output_dir <- file.path("outputs", "DEA_results", paste0(cancer_label, "_", source_name))
  if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)
  
  top_genes_file <- file.path("outputs", "DEA", paste0("top_", cancer_label, "_", source_name, "_genes.csv"))
  top_genes_df <- read.csv(top_genes_file, header = FALSE)
  top_genes_list <- trimws(unlist(strsplit(as.character(top_genes_df[1,]), ",")))
  
  volcano_plot <- create_volcano_plot(dea_results, cancer_label, source_name, top_genes_list)
  ggsave(paste0(output_dir, "/volcano_plot_with_top_genes.png"), 
         volcano_plot, width = 12, height = 10, dpi = 300)
  
  analyze_top_genes_regulation(dea_results, cancer_label, top_genes_list, output_dir)
  
  
  return(volcano_plot)
}

create_volcano_plot <- function(results, cancer_label, source_name, top_genes_list) {
  
  fdr_thresh <- 0.05
  fc_thresh <- 1.5 
  
  plot_data <- results %>%
    mutate(
      log10FDR = -log10(adj.P.Val),
      Significance = case_when(
        adj.P.Val < fdr_thresh & logFC > fc_thresh ~ "Upregulated",
        adj.P.Val < fdr_thresh & logFC < -fc_thresh ~ "Downregulated", 
        TRUE ~ "Not significant"
      )
    )
  
  n_up <- sum(plot_data$Significance == "Upregulated", na.rm = TRUE)
  n_down <- sum(plot_data$Significance == "Downregulated", na.rm = TRUE)
  
  top_genes_data <- plot_data %>%
    filter(Gene %in% top_genes_list & 
             adj.P.Val < fdr_thresh & 
             abs(logFC) > fc_thresh)
  
  base_point_size <- 1.2
  
  p <- ggplot(plot_data, aes(x = logFC, y = log10FDR, color = Significance)) +
    geom_point(alpha = 0.4, size = base_point_size, shape = 19, stroke = 0,
               position = position_jitter(width = 0.05, height = 0.05)) +
    scale_color_manual(values = c(
      "Downregulated" = "#00AFBB",   
      "Upregulated" = "#bb0c00",     
      "Not significant" = "#CCCCCC"           
    )) +
    guides(color = guide_legend(override.aes = list(size = 3))) +
    geom_hline(yintercept = -log10(fdr_thresh), linetype = "dashed", color = "darkgray", linewidth = 0.5) +
    geom_vline(xintercept = c(-fc_thresh, fc_thresh), linetype = "dashed", color = "darkgray", linewidth = 0.5) +
    labs(x = "Expression Change (log2 FC)",
         y = "Statistical Significance (-log10 FDR)") +
    theme_minimal() +
    theme(
      axis.title.x = element_text(size = 16),
      axis.title.y = element_text(size = 16),
      legend.text = element_text(size = 12),
      legend.position = "bottom",
      plot.subtitle = element_text(size = 11),
      panel.grid = element_blank(),
      legend.title = element_blank() 
    )
  
  if (nrow(top_genes_data) > 0) {
    p <- p + 
      geom_point(data = top_genes_data, 
                 aes(x = logFC, y = log10FDR), 
                 color = "black", 
                 size = base_point_size, 
                 shape = 19,       
                 alpha = 1) +
      geom_text_repel(
        data = top_genes_data,
        aes(label = Gene),
        color = "black",
        size = 3.5,
        fontface = "bold",
        max.overlaps = 30,
        box.padding = 0.3,
        point.padding = 0.8,
        segment.color = "black",
        segment.alpha = 0.4,
        show.legend = FALSE
      )
  } 
  return(p)
}

analyze_top_genes_regulation <- function(results, cancer_label, top_genes_list, output_dir) {
  
  matches <- top_genes_list %in% results$Gene
  cat("Exact matches found:", sum(matches), "/", length(top_genes_list), "\n")
  
  top_genes_results <- results[results$Gene %in% top_genes_list, ]
  if (nrow(top_genes_results) == 0) {
    cat("No top genes found in DE results after matching attempts\n")
    return(NULL)
  }
  
  cat("\n", rep("=", 70), "\n", sep = "")
  cat("TOP GENES REGULATION ANALYSIS:", cancer_label, "\n")
  cat(rep("=", 70), "\n", sep = "")
  
  fc_thresh <- 1.5
  fdr_thresh <- 0.05
  
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
  
  cat("\nSUMMARY STATISTICS:\n")
  cat("Total top genes in file:", length(top_genes_list), "\n")
  cat("Top genes found in DE results:", nrow(top_genes_results), "\n")
  
  if (nrow(top_genes_results) > 0) {
    upregulated <- sum(top_genes_results$logFC > fc_thresh)
    downregulated <- sum(top_genes_results$logFC < -fc_thresh)
    significant <- sum(top_genes_results$adj.P.Val < fdr_thresh)
    biologically_relevant <- sum(top_genes_results$adj.P.Val < fdr_thresh & 
                                   abs(top_genes_results$logFC) > fc_thresh)
    
    cat("Upregulated (logFC > 1.5):", upregulated, "\n")
    cat("Downregulated (logFC < -1.5):", downregulated, "\n")
    cat("Significant (FDR < 0.05):", significant, "\n")
    cat("Biologically relevant (FDR < 0.05 & |logFC| > 1.5):", biologically_relevant, "\n")
    
    missing_genes <- setdiff(top_genes_list, top_genes_results$Gene)
    if (length(missing_genes) > 0) {
      cat("\nMISSING GENES (not found in DE results):\n")
      print(missing_genes)
    }
  }
  
  biologically_relevant_top_genes <- top_genes_results %>%
    filter(adj.P.Val < fdr_thresh & abs(logFC) > fc_thresh) %>% 
    pull(Gene)
  
  top_genes_output <- paste0(output_dir, "/top_genes_analysis.csv")
  write.csv(top_genes_results, top_genes_output, row.names = FALSE)
  cat("Top genes analysis saved to:", top_genes_output, "\n")
  
  return(top_genes_results)
}


gbm_ge_volcano_plot <- perform_limma_dea('GBM', 'ge')
gbm_methyl_volcano_plot <- perform_limma_dea('GBM', 'methyl')

lgg_ge_volcano_plot <- perform_limma_dea('LGG', 'ge')
lgg_methyl_volcano_plot <- perform_limma_dea('LGG', 'methyl')

# Create joined images
combined_gbm_plot <- (gbm_ge_volcano_plot + gbm_methyl_volcano_plot) + 
  plot_annotation(tag_levels = 'A') & 
  theme(plot.tag = element_text(size = 40, face = "bold"),
      legend.text = element_text(size = 16)) 

combined_gbm_plot <- combined_gbm_plot + 
  plot_layout(guides = 'collect') & 
  theme(legend.position = 'bottom')

ggsave(file.path("outputs", "DEA_results", "GBM_combined_volcano.png"),
       combined_gbm_plot, width = 16, height = 8)

combined_lgg_plot <- (lgg_ge_volcano_plot + lgg_methyl_volcano_plot) + 
  plot_annotation(tag_levels = 'A') & 
  theme(plot.tag = element_text(size = 40, face = "bold"),
      legend.text = element_text(size = 16)) 

combined_lgg_plot <- combined_lgg_plot + 
  plot_layout(guides = 'collect') & 
  theme(legend.position = 'bottom')

ggsave(file.path("outputs", "DEA_results", "LGG_combined_volcano.png"),
       combined_lgg_plot, width = 16, height = 8)
