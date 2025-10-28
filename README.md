# Explainable Deep Learning for Modeling Genomic Variation and Plant Environmental Adaptation

This repository contains a complete pipeline for predicting environmental variables from genomic sequences using deep learning embeddings and machine learning models. The workflow processes raw genomic and environmental data, generates DNA sequence embeddings using foundation models, and trains predictive models to link genotype to phenotype.

## Project Overview

**Goal:** Predict bioclimatic and soil variables for *Arabidopsis thaliana* natural accessions using genomic sequences.

**Approach:**
1. Process and filter environmental and genomic data
2. Generate dense vector embeddings from DNA sequences using pretrained foundation models
3. Train machine learning models to predict environmental variables from embeddings
4. Explain model predictions through attribution analysis and mutation experiments

**Key Technologies:**
- **Foundation Models:** Caduceus (BiMamba)
- **Machine Learning:** ElasticNet, Support Vector Regression
- **Interpretation:** Integrated Gradients, MEME Suite
- **Data Sources:** 1001 Genomes, WorldClim 2.1, SoilGrids

---

## Repository Structure

```
release/
├── Data_Processing/              # Environmental and genomic data preparation
├── SparSNP_and_Selection/       # SNP selection and GWAS preprocessing
├── Embedding_and_Model_Training/ # DNA embedding generation and model training
├── Prediction_Explanation/       # Model interpretation and analysis
│   ├── attribution/             # Attribution analysis and motif discovery
│   └── mutations/               # In silico mutation experiments
└── Results_and_Analysis/        # Performance evaluation and visualization
```

---

## Pipeline Overview

### 1. Data Processing
**Directory:** `Data_Processing/`

Prepares environmental and genomic data for downstream analysis.

**Key Processes:**
- Download and merge bioclimatic variables (WorldClim 2.1: BIO1-BIO19)
- Integrate soil properties (SoilGrids: clay, pH, organic carbon, etc.)
- Filter SNPs by quality (missingness ≤5%, MAF ≥1%)
- Identify and combine highly correlated variables (|r| > 0.9)
- Create stratified train/validation/test splits using consensus clustering
- Extract genomic sequences from 1001 Genomes API

**Input:**
- Raw SNP data (`1001_snp.csv`)
- Geographic coordinates with environmental data
- Accession IDs

**Output:**
- `combined_variables_dataset_normalized.csv` - Environmental variables (normalized)
- `train_ids.txt`, `val_ids.txt`, `test_ids.txt` - Data splits
- FASTA files per accession with genomic sequences

**See:** [Data_Processing/README.md](Data_Processing/README.md)

---

### 2. SparSNP and SNP Selection
**Directory:** `SparSNP_and_Selection/`

Selects informative SNPs using sparse regression (SparSNP) for feature reduction.

**Key Processes:**
- Run SparSNP cross-validation to identify optimal SNPs
- Evaluate SNP selection for each environmental variable
- Compare SparSNP-based predictions with full genome approaches
- Generate PLINK-format files for GWAS tools

**Input:**
- Filtered PLINK files (`.bed`, `.bim`, `.fam`)
- Environmental phenotype data
- Train/validation splits

**Output:**
- Selected SNP positions per variable
- SparSNP prediction results (R², cross-validation scores)
- PLINK files filtered to selected SNPs

**See:** [SparSNP_and_Selection/README.md](SparSNP_and_Selection/README.md)

---

### 3. Embedding and Model Training
**Directory:** `Embedding_and_Model_Training/`

Generates DNA sequence embeddings and trains predictive models.

**Key Processes:**

**Phase 1: Sequence Embedding**
- Load genomic sequences (FASTA) per accession
- Generate embeddings using pretrained foundation models:
  - **Caduceus:** BiMamba architecture (131k bp context, 256-dim embeddings)
  - **Nucleotide Transformer:** BERT-based (variable length, 768-1024 dim)
  - **Enformer:** Transformer for regulatory elements (196k bp, 3072 dim)
- Apply bidirectional processing (forward + reverse complement)
- Pool embeddings (max or average pooling)

**Phase 2: Model Training**
- Train multi-output regression models:
  - **ElasticNet:** L1+L2 regularization (fast, interpretable)
  - **SVR:** Support Vector Regression (higher accuracy, slower)
- Perform hyperparameter tuning via GridSearchCV
- Evaluate on validation set (R², MSE, Spearman correlation)

**Input:**
- FASTA files with genomic sequences
- Environmental variables (normalized)
- Train/validation ID splits

**Output:**
- Embedding pickle files (`{model}_embeddings.pkl`)
- Trained models (`{model_type}_model_{embedding}.pkl`)
- Performance results (`multioutput_results.csv`)

**See:** [Embedding_and_Model_Training/README.md](Embedding_and_Model_Training/README.md)

---

### 4. Prediction Explanation
**Directory:** `Prediction_Explanation/`

Interprets model predictions through attribution analysis and mutation experiments.

#### 4A. Attribution Analysis
**Subdirectory:** `Prediction_Explanation/attribution/`

Identifies important genomic positions and sequence motifs driving predictions.

**Key Processes:**
- Extract important model weights (SNP × embedding dimension pairs)
- Generate saliency maps using Integrated Gradients
- Extract gradient peaks (high attribution positions)
- Discover enriched motifs using MEME Suite SEA
- Filter significant motifs (E-value thresholds)
- Compare motifs against known databases (JASPAR) using TomTom
- Identify non-redundant motifs via self-comparison

**Input:**
- Trained models
- Genomic sequences
- Embeddings
- Environmental predictions

**Output:**
- Weight explanations (`weights_explanation.csv`)
- Saliency maps per SNP and variable
- Gradient peak positions
- Discovered motifs (MEME format)
- Motif enrichment results (SEA output)
- Motif database matches (TomTom output)

**See:** [Prediction_Explanation/attribution/README.md](Prediction_Explanation/attribution/README.md)

---

#### 4B. Mutation Analysis
**Subdirectory:** `Prediction_Explanation/mutations/`

Evaluates prediction changes from in silico sequence mutations.

**Key Processes:**
- Identify top/bottom accessions by phenotype value
- Extract sequence motifs from extreme phenotype groups
- Generate mutated sequences at important positions
- Re-embed mutated sequences
- Predict environmental variables for mutants
- Quantify prediction changes (single mutations, combinations)

**Input:**
- Trained models
- Genomic sequences (FASTA)
- Gradient peaks or important weights
- Phenotype values

**Output:**
- Top/bottom motif comparisons (`motif_comparison.tsv`)
- Mutated sequence embeddings
- Mutation effect sizes (`best_single_mutation_per_accession.csv`)
- Combination mutation effects (`best_combo_mutation_per_accession.csv`)

**See:** [Prediction_Explanation/mutations/README.md](Prediction_Explanation/mutations/README.md)

---

### 5. Results and Analysis
**Directory:** `Results_and_Analysis/`

Analyzes model performance and visualizes results.

**Key Processes:**
- Compare model performance across configurations:
  - Different foundation models (Caduceus, NT, Enformer)
  - Sequence length experiments (512bp, 1kb, 100kb)
  - Training set size experiments
  - SparSNP vs. full genome comparison
- Visualize embeddings using dimensionality reduction (t-SNE, UMAP, PCA)
- Generate publication-quality plots
- Compute bootstrap confidence intervals
- Statistical significance testing

**Input:**
- Model prediction results (CSV files)
- Embeddings pickle files
- Metadata (Köppen-Geiger climate zones, coordinates)

**Output:**
- Performance comparison plots
- Bootstrap confidence intervals
- Statistical test results
- Embedding visualizations (2D projections)
- Summary tables (mean R², Spearman across variables)

**See:** [Results_and_Analysis/README.md](Results_and_Analysis/README.md)

---

## Quick Start Guide

### Prerequisites

**Software:**
- Python 3.8+
- R 3.6+ (for SparSNP)
- PLINK 1.9+
- Singularity (for containerized execution)
- SLURM (for cluster job scheduling)

**Python Packages:**
```bash
pip install numpy pandas scikit-learn scipy matplotlib seaborn
pip install torch transformers biopython tqdm
pip install geopandas cartopy rasterio  # For geographic data
```

**External Tools:**
- MEME Suite (motif analysis)
- SparSNP (SNP selection)

---

### End-to-End Workflow

```bash
# 1. Data Processing
cd Data_Processing/
jupyter notebook get_data.ipynb           # Download and merge data
python filter_and_save.py                 # Filter SNPs
python correlate_and_cluster_variables.py # Analyze correlations
python combine_correlated_variables.py    # Combine variables
python split_data.py                      # Create train/val/test splits

# 2. SNP Selection (optional - for comparison)
cd ../SparSNP_and_Selection/
python make_files.py                      # Prepare PLINK files
bash run_sparsnp.sh                       # Run SparSNP
Rscript get_results.R                     # Analyze results

# 3. Embedding Generation
cd ../Embedding_and_Model_Training/
sbatch run_embed_caduceus.sh              # Generate embeddings (GPU)

# 4. Model Training
sbatch run_train_predict_models.sh        # Train models (CPU)

# 5. Attribution Analysis
cd ../Prediction_Explanation/attribution/
sbatch run_explain_model_weights.sh       # Extract important weights
sbatch run_saliency_maps.sh               # Generate saliency maps
sbatch run_extract_gradient_peaks.sh      # Find peak positions
sbatch memesuite_sea.sh                   # Motif enrichment
sbatch filter_significant_motifs.sh       # Filter motifs
sbatch memesuite_tomtom.sh                # Compare motifs

# 6. Mutation Analysis (optional)
cd ../mutations/
sbatch run_extract_top_bottom_motif.sh    # Extract extreme phenotypes
sbatch run_embed_caduceus_mutations.sh    # Embed mutations
sbatch run_analyze_mutation_effects.sh    # Analyze effects

# 7. Analyze Results
cd ../../Results_and_Analysis/
jupyter notebook thesis_plotting.ipynb     # Performance analysis
jupyter notebook visualize_embedding.ipynb # Embedding visualization
```

---

## Data Flow Diagram

```
Raw Data
├── 1001 Genomes (SNPs, sequences)
├── WorldClim 2.1 (bioclimatic variables)
└── SoilGrids (soil properties)
         ↓
    [Data Processing]
         ↓
Environmental Data + Filtered Genomic Sequences
         ↓
    [SparSNP Selection]
         ↓
Selected SNP Positions (top informative SNPs)
         ↓
Extract Genomic Regions Around Selected SNPs
         ↓
    [Embedding Generation]
         ↓
DNA Embeddings from Selected Regions
         ↓
    [Model Training]
         ↓
Trained Models + Predictions
         ↓
    ┌────────────────┴────────────────┐
    ↓                                 ↓
[Attribution Analysis]          [Mutation Analysis]
    ↓                                 ↓
Important Positions              Mutation Effects
Enriched Motifs                  Phenotype Changes
    ↓                                 ↓
    └─────────→ [Results Analysis] ←─────────┘
                     ↓
    Performance Metrics + Visualizations
```

---

## File Organization

### Input Data (Not Included)

The following data files are required but not included in the repository:

```
Data/
├── 1001_snp.csv                          # Raw SNP genotype matrix (1001 Genomes)
├── coords_with_bioclim_30s_fixed.csv     # Bioclimatic variables per accession
├── coords_with_soil.csv                  # Soil properties per accession
├── columnIDs.txt                         # List of accession IDs
└── Pseudogenomes/                        # FASTA files per accession
    ├── 6909.fa
    ├── 9728.fa
    └── ...
```

**Data Sources:**
- **1001 Genomes:** https://1001genomes.org/
- **WorldClim 2.1:** https://www.worldclim.org/
- **SoilGrids:** https://www.isric.org/explore/soilgrids

### Output Structure

```
Results/
├── Embeddings/                           # Generated embeddings
│   ├── Caduceus_512.pkl
│   ├── Caduceus_1000.pkl
│   └── ...
├── Models/                               # Trained models
│   ├── elasticnet_model_Caduceus_512.pkl
│   ├── svr_model_Caduceus_1000.pkl
│   └── ...
├── Attribution/                          # Attribution analysis results
│   ├── weights_explanation.csv
│   ├── gradient_peaks/
│   ├── motifs/
│   └── tomtom/
├── Mutations/                            # Mutation experiment results
│   └── AnalysisAll/
│       ├── Positive/
│       └── Negative/
└── Performance/                          # Performance metrics
    ├── combined_r2_results.csv
    ├── combined_spearman_results.csv
    └── plots/
```

---

## Computational Requirements

### Minimum Requirements
- **Data Processing:** 32 GB RAM, 4 CPU cores
- **SNP Selection:** 64 GB RAM, 8 CPU cores
- **Embedding Generation:** 1 GPU (40 GB VRAM), 128 GB RAM
- **Model Training:** 128 GB RAM, 16 CPU cores
- **Attribution Analysis:** 64 GB RAM, 8 CPU cores

### Recommended Requirements
- **Full Pipeline:** HPC cluster with GPU nodes
- **Storage:** 500 GB for intermediate files
- **Runtime:** ~3-5 days for complete analysis (1000 accessions)

---

## Contributing

This pipeline is designed for research purposes. For questions or issues:

1. Check the README in the relevant subdirectory
2. Review troubleshooting sections
3. Verify data format compatibility
4. Ensure computational resources are sufficient

---

## Acknowledgments

- **1001 Genomes Consortium** - Genomic data
- **HuggingFace** - Foundation model hosting
- **MEME Suite** - Motif analysis tools
- **Kuleshov Lab** - Caduceus model
- **InstaDeep** - Nucleotide Transformer
- **DeepMind** - Enformer model

---

## Directory Details

For detailed documentation of each component:

1. **[Data_Processing/README.md](Data_Processing/README.md)** - Data acquisition, filtering, and preprocessing
2. **[SparSNP_and_Selection/README.md](SparSNP_and_Selection/README.md)** - SNP selection and GWAS preparation
3. **[Embedding_and_Model_Training/README.md](Embedding_and_Model_Training/README.md)** - Embedding generation and model training
4. **[Prediction_Explanation/attribution/README.md](Prediction_Explanation/attribution/README.md)** - Attribution analysis and motif discovery
5. **[Prediction_Explanation/mutations/README.md](Prediction_Explanation/mutations/README.md)** - Mutation experiments and effect analysis
6. **[Results_and_Analysis/README.md](Results_and_Analysis/README.md)** - Performance evaluation and visualization

