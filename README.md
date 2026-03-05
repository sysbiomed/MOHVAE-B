# MOHVAE-B: Multi-Omics Hierarchical Variational Autoencoder with Bayesian Networks for Biomarker Discovery

[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## Overview

MOHVAE-B is a comprehensive deep learning framework designed for integrative analysis of multi-omics data and biomarker discovery in cancer research. The framework combines a Hierarchical Variational Autoencoder (H-VAE) with SHAP interpretability and Bayesian Networks to identify and validate molecular features associated with cancer subtypes.

Key features:
- **Multi-omics integration**: Simultaneously processes DNA methylation, gene expression, and protein expression data (which can be extended to other omics modalities).
- **Interpretability**: SHAP values for feature importance ranking.
- **Biological validation**: Differential expression analysis and DNA methylation profiling.
- **Dependency modeling**: Bayesian Networks to capture relationships between biomarkers.

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Fredd124/mscthesis_24-25_MEIC.git
cd mscthesis_24-25_MEIC
```

2. Install the requirements:
```bash
pip install -r requirements.txt
``` 

3. Install R alongside the necessay packages:
```r
install.packages(c("limma", "ggplot2", "dplyr", "ggrepel", "patchwork"))
if (!require("BiocManager", quietly = TRUE))
    install.packages("BiocManager")
BiocManager::install(c(
    "clusterProfiler",
    "org.Hs.eg.db",
    "AnnotationDbi",
    "enrichplot",
    "DOSE"
))
```

4. Install [GeNIe](https://www.bayesfusion.com/genie/) for Bayesian Network visualization.

GeNIe is a graphical user interface for Bayesian Networks developed by BayesFusion. Although other softwares can be used, GeNIe was chosen due to its freely available for academic use. 


## Downloading TCGA Data

For this work, TCGA (The Cancer Genome Atlas) data was used. The data was downloaded via the [UCSC Xena platform](https://xenabrowser.net/datapages/?hub=https://tcga.xenahubs.net:443), which provides precompiled TCGA datasets. The file [`download_UCSCtcga_data.ipynb`](download_UCSCtcga_data.ipynb) reproduces all the steps to download and preprocess all the data used in this project. Note that a substantial amount of memory is needed to be able to save all the files downloaded and generated in the data preprocessing steps. 

## Model Optimization

Before running the main project, an optimization step was applied to find the best hyperparameters for the H-VAE model. The file [`H-VAE/bayesian_hyperparam_search.py`](H-VAE/bayesian_hyperparam_search.py) reproduces all the necessary steps using [Optuna](https://optuna.org/), a hyperparameter optimization framework. The optimization process was conducted in two stages:

1. **First Stage**: Optimization of training parameters (dropout, batch size, learning rate, regularization weights), using the parameters from the original model.
2. **Second Stage**: Optimization of architectural dimensions (encoder/decoder layers, latent space size) using the optimal training parameters found in stage one.

To run the optimization and choose the number of trials (which is set to 30 by default):
```bash
python H-VAE/bayesian_hyperparam_search.py -n 20
```

Note that this process is time consuming depending on the GPU used. For most cases 20 trials is an acceptable number.

## Running the Framework

The file [`H-VAE/main_framework.ipynb`](H-VAE/main_framework.ipynb) is the main file of this project. This notebook reproduces all the steps for running the framework and displays the results from our original work. 

## License

This source code is licensed under the [`MIT`](LICENSE) license.