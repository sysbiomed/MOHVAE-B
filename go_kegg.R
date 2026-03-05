library(clusterProfiler)
library(org.Hs.eg.db)
library(AnnotationDbi)
library(enrichplot)
library(DOSE)
base_dir <- dirname(rstudioapi::getActiveDocumentContext()$path)
setwd(base_dir)

perform_go_kegg_analysis <- function(cancer_label, source_name, manual_list_genes = NULL) {
  output_dir <- file.path("outputs", "DEA_results", paste0(cancer_label, "_", source_name))
  if (!dir.exists(output_dir)) {
    dir.create(output_dir, recursive = TRUE)
  }
  
  if (is.null(manual_list_genes)) {
    top_genes_file <- file.path("outputs", "DEA", paste0("top_", cancer_label, "_", source_name, "_genes.csv"))
    top_genes_df <- read.csv(top_genes_file, header = FALSE)
    top_genes_list <- unlist(strsplit(as.character(top_genes_df[1,]), ","))
    top_genes_list <- trimws(top_genes_list)
  }
  else {
    top_genes_list <- manual_list_genes
    output_dir <- file.path("outputs", "BNs", paste0(cancer_label, "_", source_name))
    if (!dir.exists(output_dir)) {
      dir.create(output_dir, recursive = TRUE)
    }
  }
  
  if (length(top_genes_list) < 2) {
    cat("Too few top genes found in results:", length(top_genes_list), "\n")
    return(NULL)
  }
  
  gene_symbols <- top_genes_list
  entrez_ids <- mapIds(org.Hs.eg.db,
                       keys = gene_symbols,
                       column = "ENTREZID",
                       keytype = "SYMBOL",
                       multiVals = "first")
  entrez_ids <- na.omit(entrez_ids)
  
  if (length(entrez_ids) < 2) {
    cat("Too few genes with Entrez IDs for enrichment analysis\n")
    return(NULL)
  }
  
  cat("Analyzing", length(entrez_ids), "significant genes with Entrez IDs\n")
  
  go_bp <- enrichGO(gene = entrez_ids,
                    OrgDb = org.Hs.eg.db,
                    keyType = "ENTREZID",
                    ont = "BP",  
                    pAdjustMethod = "BH",
                    pvalueCutoff = 0.05,
                    qvalueCutoff = 0.1,
                    readable = TRUE)
  
  kegg <- enrichKEGG(gene = entrez_ids,
                     organism = 'hsa',  
                     pvalueCutoff = 0.05,
                     qvalueCutoff = 0.1)
  
  # If results, save
  if (!is.null(go_bp) && nrow(go_bp) > 0) {
    write.csv(go_bp, paste0(output_dir, "/GO_BP_enrichment.csv"), row.names = FALSE)
    cat("GO Biological Process results saved\n")
  }
  
  
  if (!is.null(kegg) && nrow(kegg) > 0) {
    kegg_res <- as.data.frame(kegg)
    kegg_res$GeneSymbol <- mapIds(org.Hs.eg.db,
                                  keys = as.character(kegg_res$geneID),
                                  column = "SYMBOL",
                                  keytype = "ENTREZID",
                                  multiVals = "first")
    write.csv(kegg_res, paste0(output_dir, "/KEGG_pathway_enrichment.csv"), row.names = FALSE)
    cat("KEGG Pathway results saved\n")
  }
  
  cat("\n=== TOP ENRICHED TERMS ===\n")
  
  if (!is.null(go_bp) && nrow(go_bp) > 0) {
    cat("\nTop 5 GO Biological Processes:\n")
    print(head(go_bp[, c("Description", "p.adjust", "Count")], 5))
  }
  
  if (!is.null(kegg) && nrow(kegg) > 0) {
    cat("\nTop 5 KEGG Pathways:\n")
    print(head(kegg[, c("Description", "p.adjust", "Count")], 5))
  }
}

# if want to use for a specific set (ex: branch of the BNs)
#list <- c("FKBP1C", "NACA2", "SCRG1", "PMP2", "MIR9-1HG", "SCHIP1", "AQP4")
#perform_go_kegg_analysis('GBM', 'ge', list)
perform_go_kegg_analysis('GBM', 'ge')
perform_go_kegg_analysis('LGG', 'ge')

perform_go_kegg_analysis('GBM', 'methyl')
perform_go_kegg_analysis('LGG', 'methyl')
