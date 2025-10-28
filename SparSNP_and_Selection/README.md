# SparSNP Pipeline: From PLINK to Selected SNPs

This document describes the complete process for running SparSNP (Sparse SNP selection) on genomic data to identify key SNPs for predicting bioclimatic and soil variables.

## Overview

The pipeline takes filtered PLINK files and phenotype data, runs SparSNP cross-validation to select optimal SNPs, and evaluates predictions on a test set. The process identifies the most informative genetic markers for each environmental variable.

## Prerequisites

### Required Software
- **PLINK** (v1.9 or later) - for genotype file manipulation
- **SparSNP** - Sparse regression tool for SNP selection (collection of shell and R scripts)
  - Download from: https://github.com/medical-genomics-group/SparSNP
  - Install scripts to your PATH or specify full paths in `run_sparsnp.sh`
- **R** (v3.6 or later) - for running SparSNP scripts
- **Python** (v3.7 or later) - for data preparation

### Required R Packages

The analysis scripts in this repository use base R functions. SparSNP itself may require additional packages - check the SparSNP documentation for its specific dependencies.

For the result compilation script (`get_results.R`):
```r
# No additional packages required - uses base R only
```

**Note:** SparSNP itself is not an R package. It's a collection of shell scripts (`crossval.sh`, `predict.sh`) and R scripts (`eval.R`, `getmodels.R`, `evalprofile.R`) that must be in your PATH or called with full paths.

### Required Python Packages
```bash
pip install pandas numpy
```

## Input Data Requirements

### 1. PLINK Files

**IMPORTANT: Pre-filtering Required**

Your original PLINK files must be filtered for quality before running SparSNP:

```bash
# Filter by missingness (5%) and minor allele frequency (1%)
plink --bfile raw_data \
      --geno 0.05 \
      --maf 0.01 \
      --make-bed \
      --out filtered_data
```

**Filter criteria:**
- `--geno 0.05`: Remove SNPs with >5% missing genotype calls
- `--maf 0.01`: Remove SNPs with minor allele frequency <1%

These filters ensure data quality and reduce computational burden while retaining informative markers.

**Required PLINK file format:**
- `.bed` - Binary genotype file
- `.bim` - SNP information file (chromosome, SNP ID, position, alleles)
- `.fam` - Sample information file (family ID, individual ID, phenotype)

### 2. Phenotype Data

Phenotype data should be in a CSV file with normalized environmental variables:
- `combined_variables_dataset_normalized.csv`

**Columns:**
- `IID` - Individual ID (must match PLINK `.fam` file)
- `FID` - Family ID
- `LONG`, `LAT` - Geographic coordinates (optional)
- Environmental variables (e.g., `BIO1`, `BIO_11_6`, `clay_mean`, etc.)

**Note:** Variables should be normalized (Z-score or Min-Max) before use.

## Pipeline Steps

### Step 1: Data Preparation

Split your data into training, validation, and test sets, then prepare PLINK files for each variable.

**1.1. Create train/val/test splits**

Create ID lists for each split:
```bash
# train_ids.txt - Training sample IDs
# val_ids.txt - Validation sample IDs  
# test_ids.txt - Test sample IDs
```

**1.2. Split PLINK files by dataset**

```bash
# Extract training set
plink --bfile filtered_data \
      --keep train_ids.txt \
      --make-bed \
      --out Plink/plink_train

# Extract validation set
plink --bfile filtered_data \
      --keep val_ids.txt \
      --make-bed \
      --out Plink/plink_val

# Extract test set
plink --bfile filtered_data \
      --keep test_ids.txt \
      --make-bed \
      --out Plink/plink_test
```

**1.3. Prepare variable-specific PLINK files**

Run `make_files.py` to create PLINK files for each environmental variable:

```bash
python make_files.py
```

This script:
- Reads the normalized phenotype data (`combined_variables_dataset_normalized.csv`)
- Creates separate PLINK files for each variable (updates `.fam` files with phenotype values)
- Outputs files to `Datasets/JointVariables/{train,val,test}/plink_<variable>.{bed,bim,fam}`

**Directory structure after preparation:**
```
Datasets/JointVariables/
├── train/
│   ├── plink_BIO1.{bed,bim,fam}
│   ├── plink_BIO_11_6.{bed,bim,fam}
│   └── ...
├── val/
│   └── ...
└── test/
    └── ...
```

### Step 2: Run SparSNP Cross-Validation

**2.1. Execute the SparSNP pipeline**

```bash
cd Datasets
bash run_sparsnp.sh JointVariables
```

**What this does:**
- For each variable, runs 5-fold cross-validation on the training set
- Tests multiple sparsity levels (number of SNPs to select)
- Identifies the optimal number of SNPs based on cross-validation performance
- Extracts the best model's SNPs
- Generates predictions on the test set
- Evaluates prediction accuracy (R² and Spearman correlation)

**2.2. SparSNP stages:**

1. **Discovery (Cross-Validation):**
   ```bash
   NUMPROCS=5 DIR="JointVariables/discoveryNorm/<variable>" crossval.sh plink_<variable> linear
   ```
   - Runs cross-validation to find optimal sparsity
   - Output: `JointVariables/discoveryNorm/<variable>/`

2. **Model Selection:**
   ```bash
   eval.R dir="JointVariables/discoveryNorm/<variable>"
   ```
   - Identifies best number of SNPs
   - Output: Printed to console and saved to `best_model_snps_all.txt`

3. **Extract Top SNPs:**
   ```bash
   getmodels.R nzreq=<best_snps> dir="JointVariables/discoveryNorm/<variable>"
   ```
   - Extracts the top SNPs at optimal sparsity
   - Output: `topsnps.txt`, `avg_weights_opt.score`

4. **Prediction:**
   ```bash
   OUTDIR="JointVariables/predictNorm/<variable>" DIR="JointVariables/discoveryNorm/<variable>" predict.sh plink_<variable>_test
   ```
   - Applies trained model to test set
   - Output: `JointVariables/predictNorm/<variable>/`

5. **Evaluation:**
   ```bash
   evalprofile.R model=linear indir="JointVariables/discoveryNorm/<variable>" outdir="JointVariables/predictNorm/<variable>"
   ```
   - Calculates prediction accuracy
   - Output: `results.RData`

### Step 3: Compile Results

**3.1. Extract consensus predictions and performance metrics**

```bash
cd Datasets/JointVariables
Rscript ../get_results.R
```

This script:
- Loads predictions for each variable from `predict/<variable>/results.RData`
- Calculates consensus predictions (average across cross-validation folds)
- Computes R² and Spearman correlation for each variable
- Saves results to `prediction/consensus_results.txt`

**Outputs:**
- `prediction/consensus_pred_<variable>.csv` - Consensus predictions per variable
- `prediction/prediction_errors_<variable>.csv` - Prediction errors per sample
- `prediction/consensus_results.txt` - Summary table of R² and Spearman coefficients

### Step 4: Analyze Selected SNPs

**4.1. Extract and visualize selected SNPs**

```bash
python ../../get_snps.py
```

This script:
- Reads `best_model_snps_all.txt` to find optimal SNP counts
- Extracts top SNPs from each variable's `topsnps.txt`
- Identifies genomic regions (windows) enriched for selected SNPs
- Creates clustering visualizations of variables based on SNP sharing

**Outputs:**
- `all_snps.txt` - All unique SNPs selected across variables
- `snps_variable_clusters.csv` - Hierarchical clustering of variables by SNP weights
- `snp_weight_matrix.csv` - Matrix of variables × SNPs with SparSNP weights
- Plots in `Plots/Report/Clustering/` - Heatmaps showing SNP patterns

## Output Files

### Key Output Files

| File | Location | Description |
|------|----------|-------------|
| `best_model_snps_all.txt` | `JointVariables/` | Optimal SNP count per variable |
| `topsnps.txt` | `discoveryNorm/<var>/` | Top SNPs for each variable |
| `avg_weights_opt.score` | `discoveryNorm/<var>/` | SNP weights (effect sizes) |
| `consensus_results.txt` | `prediction/` | R² and Spearman per variable |
| `consensus_pred_<var>.csv` | `prediction/` | Predictions per sample |
| `all_snps.txt` | Root | All unique SNPs across variables |
| `snp_weight_matrix.csv` | Root | Variables × SNPs weight matrix |

### Directory Structure (After Complete Run)

```
Datasets/JointVariables/
├── train/
│   └── plink_<variable>.{bed,bim,fam}
├── val/
│   └── plink_<variable>.{bed,bim,fam}
├── test/
│   └── plink_<variable>.{bed,bim,fam}
├── discovery/
│   └── <variable>/
│       ├── topsnps.txt
│       ├── avg_weights_opt.score
│       └── cv_fold_*.RData
├── predict/
│   └── <variable>/
│       └── results.RData
├── prediction/
│   ├── consensus_results.txt
│   ├── consensus_pred_<variable>.csv
│   └── prediction_errors_<variable>.csv
└── best_model_snps_all.txt
```

## Variables Processed

The pipeline processes both combined and individual variables from the normalized dataset:

**Combined variables** (from highly correlated clusters):
- `BIO_4_7` - Temperature Seasonality & Annual Range
- `BIO_10_5` - Max & Mean Temperature of Warmest Period
- `BIO_11_6` - Min & Mean Temperature of Coldest Period
- `BIO_12_13_16` - Annual Precipitation & Wettest Periods
- `BIO_14_17` - Precipitation of Driest Period
- `Soil_Nitrogen_SOC` - Soil Nitrogen & Organic Carbon

**Individual variables:**
- `BIO1`, `BIO2`, `BIO3`, `BIO8`, `BIO9`, `BIO15`, `BIO18`, `BIO19` - Bioclimatic variables
- `clay_mean`, `bdod_mean`, `wv0033_mean`, `phh2o_mean` - Soil variables

## Customization

### Adjusting Variables

Edit the `variables` array in `run_sparsnp.sh`:
```bash
variables=('BIO1' 'BIO2' 'custom_var')
```

Also update `get_results.R`:
```r
variables <- c('BIO1', 'BIO2', 'custom_var')
```

And `make_files.py` to match your phenotype CSV columns.

### Changing Cross-Validation Folds

Modify `NUMPROCS` in `run_sparsnp.sh`:
```bash
NUMPROCS=10  # Use 10-fold CV instead of 5-fold
```

### Filtering Thresholds

Adjust PLINK quality filters:
```bash
# More stringent filters
plink --bfile raw_data --geno 0.01 --maf 0.05 --make-bed --out filtered_data

# More lenient filters
plink --bfile raw_data --geno 0.10 --maf 0.005 --make-bed --out filtered_data
```

**Recommendations:**
- `--geno 0.05`: Standard for most studies (removes SNPs with >5% missingness)
- `--maf 0.01`: Removes very rare variants that lack statistical power
- For small sample sizes, consider `--maf 0.05`
- For very large datasets, `--geno 0.01` for stricter quality control

## Performance Metrics

### R² (Coefficient of Determination)
- Measures proportion of variance explained
- Range: 0 to 1 (higher is better)
- Values >0.3 indicate good predictive power for complex traits

### Spearman Correlation
- Measures rank-order correlation between predicted and observed
- Range: -1 to 1 (closer to ±1 is better)
- Robust to outliers and non-linear relationships