# Data Processing Pipeline

This directory contains scripts and notebooks for processing environmental and genomic data, from raw data acquisition to final train/validation/test splits. The pipeline handles bioclimatic variables, soil properties, and genomic sequence extraction for *Arabidopsis thaliana* accessions.

## Pipeline Overview

The data processing workflow consists of the following major steps:

1. **Data Acquisition** - Download bioclimatic and soil data from remote sources
2. **Variable Correlation Analysis** - Identify highly correlated environmental variables
3. **Variable Combination** - Merge correlated variables to reduce dimensionality
4. **Data Splitting** - Create stratified train/validation/test splits using consensus clustering
5. **Distribution Validation** - Verify split quality and variable distributions
6. **Sequence Extraction** - Download genomic sequences from 1001 Genomes API

## Quick Start

```bash
# 1. Download and merge environmental data (Jupyter notebook)
jupyter notebook get_data.ipynb

# 2. Filter SNP data by quality thresholds
python filter_and_save.py

# 3. Analyze variable correlations and identify clusters
python correlate_and_cluster_variables.py

# 4. Combine highly correlated variables
python combine_correlated_variables.py

# 5. Create stratified train/val/test splits
python split_data.py

# 6. Visualize split distributions
python plot_splits.py

# 7. Compare distributions before/after combination
python compare_variable_distributions.py
```

## Detailed Workflow

### Step 1: Data Acquisition and Merging

**Notebook:** `get_data.ipynb`

Downloads and integrates multiple data sources for *Arabidopsis thaliana* natural accessions.

**Data Sources:**

1. **Bioclimatic Variables (WorldClim 2.1)**
   - 19 bioclimatic variables (BIO1-BIO19)
   - 30 arc-second (~1 km²) resolution
   - Temperature and precipitation metrics
   - Source: `coords_with_bioclim_30s_fixed.csv`

2. **Soil Properties (SoilGrids)**
   - Clay content (`clay_mean`)
   - Bulk density (`bdod_mean`)
   - Volumetric water content (`wv0033_mean`)
   - Soil pH (`phh2o_mean`)
   - Soil organic carbon (`soc_mean`)
   - Nitrogen content (`nitrogen_mean`)
   - Includes uncertainty estimates for each variable
   - Source: `coords_with_soil.csv`

3. **Genomic Sequences (1001 Genomes)**
   - Pseudogenome sequences per accession
   - Region-based extraction around SNP positions
   - API: `https://tools.1001genomes.org/api/v1/pseudogenomes/`

**Key Features:**
- Merges bioclimatic and soil data on geographic coordinates (LONG, LAT)
- Handles missing data (`-3.4e+38` sentinel values)
- Creates normalized versions (Z-score standardization)
- Generates train/val/test ID files for PLINK and general use
- Downloads genomic regions around SparSNP-selected SNPs

**Output:**
- `environmental_variables.csv` - Merged bioclimatic + soil data
- `train_ids.txt`, `val_ids.txt`, `test_ids.txt` - Tab-separated ID lists
- `train_ids_plink.txt`, `val_ids_plink.txt`, `test_ids_plink.txt` - PLINK format (FID IID)
- `environmental_scaler.pkl` - StandardScaler for inverse transformation

**Normalization:**
```python
# Z-score normalization (mean=0, std=1)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
normalized_data = scaler.fit_transform(data[numeric_columns])
```

**Genomic Sequence Extraction:**

The notebook includes functions to extract genomic regions from the 1001 Genomes database:

- **Fixed-length windows:** Centers each region on a SNP position with exact window size
- **Merged regions:** Combines overlapping windows to reduce redundancy
- **Batched requests:** Handles API rate limiting and large region sets
- **Multiple window sizes:** Supports different context sizes (512bp, 1000bp, etc.)

---

### Step 2: SNP Filtering

**Script:** `filter_and_save.py`

Filters raw SNP data by quality thresholds to retain high-confidence variants.

**Input:**
- `1001_snp.csv` - Raw SNP genotype matrix from 1001 Genomes

**Filtering Criteria:**

1. **Missingness threshold:** ≤5% missing genotypes per SNP
   ```python
   missingness = np.sum(snp_genotypes == 3, axis=1) / snp_genotypes.shape[1]
   filtered_snps = snp_genotypes[missingness <= 0.05]
   ```

2. **Missing genotype imputation:** Replace with most frequent genotype
   ```python
   most_frequent_genotype = np.argmax(np.bincount(filtered_snps[i][filtered_snps[i] != 3]))
   filtered_snps[i][filtered_snps[i] == 3] = most_frequent_genotype
   ```

3. **Minor Allele Frequency (MAF) thresholds:**
   - MAF ≥ 5% → `filtered_snps_5.npy`
   - MAF ≥ 10% → `filtered_snps_10.npy`
   - MAF ≥ 15% → `filtered_snps_15.npy`

**Output:**
- `filtered_snps_{5,10,15}.npy` - Filtered genotype matrices
- `positions_{5,10,15}.npy` - Corresponding SNP positions (CHROM, POS, INDEL)

**MAF Calculation:**
```python
# Allele frequency = sum of alleles / (2 × number of individuals)
allele_freqs = np.sum(filtered_snps, axis=1) / (filtered_snps.shape[1] * 2)
filtered_snps_5 = filtered_snps[allele_freqs >= 0.05]
```

---

### Step 3: Variable Correlation Analysis

**Script:** `correlate_and_cluster_variables.py`

Identifies highly correlated environmental variables using hierarchical clustering and correlation thresholds.

**Input:**
- `coords_with_bioclim_30s_fixed.csv` - Bioclimatic data
- `coords_with_soil.csv` - Soil data
- `train_ids_plink.txt` - Training accession IDs

**Methods:**

**3.1. Correlation-based clustering:**
- Compute Pearson correlation matrix for all variable pairs
- Hierarchical clustering with Ward linkage
- Optimal leaf ordering for dendrogram visualization

**3.2. High correlation cluster identification:**

Two complementary approaches:

1. **Connected Components (Threshold: |r| > 0.9)**
   - Hard threshold on absolute correlation
   - Graph-based clustering via DFS
   - Guarantees all pairs within cluster have |r| > 0.9

2. **Agglomerative Clustering (Distance-based)**
   - Distance = √(2 × (1 - |r|))
   - Distance threshold equivalent to r = 0.9
   - Soft threshold allowing intermediate correlations

**Key Features:**
- Handles both positive and negative correlations
- Creates cluster color bars for heatmap visualization
- Prettified variable names for readability
- Comparison between clustering methods
- Per-cluster correlation statistics

**Output:**
- `correlation_matrix.csv` - Full Pearson correlation matrix
- `high_correlation_clusters.csv` - Connected components clusters
- `agglomerative_correlation_clusters.csv` - Agglomerative clustering results
- `correlation_variable_clusters.csv` - Ward clustering with N=18 clusters
- `correlation_variable_order.csv` - Dendrogram ordering
- `variable_analysis_structure.csv` - Analysis units (clusters + individual vars)
- `clustering_method_comparison.csv` - Method comparison summary
- `Plots/variable_correlation_heatmap.png` - Clustered correlation heatmap

**Typical High-Correlation Clusters:**

| Cluster ID | Variables | Description |
|------------|-----------|-------------|
| BIO_11_6 | BIO11, BIO6 | Min & Mean Temperature of Coldest Period |
| BIO_4_7 | BIO4, BIO7 | Temperature Seasonality & Annual Range |
| BIO_10_5 | BIO10, BIO5 | Max & Mean Temperature of Warmest Period |
| BIO_14_17 | BIO14, BIO17 | Precipitation of Driest Period |
| BIO_12_13_16 | BIO12, BIO13, BIO16 | Annual Precipitation & Wettest Periods |
| Soil_Nitrogen_SOC | nitrogen_mean, soc_mean | Soil Nitrogen & Organic Carbon |

**Visualization:**

The script generates a publication-quality clustermap with:
- Hierarchical clustering dendrogram (removed for clarity)
- Row and column color bars indicating cluster membership
- Correlation coefficients annotated in cells
- Diverging colormap (vlag) centered at zero
- Y-axis labels on right side to avoid overlap

---

### Step 4: Variable Combination

**Script:** `combine_correlated_variables.py`

Combines highly correlated variables into single representative variables per cluster.

**Input:**
- `high_correlation_clusters.csv` - Cluster assignments from Step 3
- `coords_with_bioclim_30s_fixed.csv` - Bioclimatic data
- `coords_with_soil.csv` - Soil data

**Combination Strategy:**

1. **Normalization (per variable):**
   ```python
   # Z-score normalization before averaging
   cluster_data_norm = (cluster_data - cluster_data.mean()) / cluster_data.std()
   ```

2. **Averaging:**
   ```python
   # Mean of normalized variables
   combined_normalized = cluster_data_norm.mean(axis=1)
   ```

3. **Individual variables:**
   - Variables not in any cluster are retained individually
   - Also normalized using same method

**Normalization Methods:**

- `zscore` (default): Mean=0, Std=1
- `minmax`: Scale to [0, 1] range
- `robust`: Median and IQR-based (outlier-resistant)
- `none`: No normalization (not recommended)

**Cluster Naming Convention:**

The script creates meaningful names for combined variables:

- **BIO-only clusters:** `BIO_4_7` (if ≤3 variables), `BIO_4to7_group` (if >3)
- **Soil-only clusters:** `Soil_Nitrogen_SOC`, `Soil_Mixed`
- **Mixed clusters:** `Mixed_BIO2_Soil1`
- **Fallback:** `Combined_6vars`

**Output:**
- `combined_variables_dataset_normalized.csv` - Final dataset with combined variables
- `variable_combination_mapping.csv` - Original → Final variable mapping

**Example Mapping:**

| Original_Variable | Final_Variable | Type | Cluster_ID | Cluster_Size |
|-------------------|----------------|------|------------|--------------|
| BIO11 | BIO_11_6 | Combined | 0 | 2 |
| BIO6 | BIO_11_6 | Combined | 0 | 2 |
| BIO1 | BIO1 | Individual | NaN | 1 |
| clay_mean | clay_mean | Individual | NaN | 1 |

**Data Reduction Summary:**

Typical reduction (depends on correlation structure):
- Original: ~32 variables (19 BIO + 6 soil + others)
- Combined: ~24 variables (6 clusters + 18 individual)
- Reduction: ~25% fewer variables while preserving information

---

### Step 5: Data Splitting with Consensus Clustering

**Script:** `split_data.py`

Creates stratified train/validation/test splits using consensus clustering to ensure representative sampling across environmental gradients.

**Input:**
- `coords_with_bioclim_30s_fixed.csv` - Bioclimatic data
- `coords_with_soil.csv` - Soil data
- Predefined splits (optional): `splits_all.json`

**Consensus Clustering Method:**

**Algorithm:** ConsensusCluster (Monti et al. 2003)
- **Resampling:** 100 iterations with 50% subsample
- **Cluster range:** K=25 to K=50
- **Base algorithm:** KMeans with 5 random initializations
- **Consensus:** Hierarchical clustering on co-occurrence matrix

```python
from consensusClustering import ConsensusCluster

consensusCluster = ConsensusCluster(
    cluster=KMeans,
    L=25,          # Minimum clusters
    K=50,          # Maximum clusters
    H=100,         # Resampling iterations
    resample_proportion=0.5
)
consensusCluster.fit(data, verbose=True)
cluster_labels = consensusCluster.predict()
```

**Stratified Splitting Logic:**

1. **Cluster-wise splitting:**
   - Each environmental cluster split independently
   - Proportional representation in train/val/test

2. **Duplicate handling:**
   - Multiple accessions from same geographic location
   - Counted and split as units to prevent data leakage

3. **Minimum sample constraints:**
   - Clusters with <2 unique samples: skip validation split
   - Clusters with <3 unique samples: skip training split
   - Ensures minimum diversity per split

4. **Split proportions:**
   - Test: 20% × 90% = 18% of total
   - Validation: 10% = 10% of total
   - Training: ~72% of total

**Key Features:**
- Geographic stratification prevents spatial autocorrelation
- Handles duplicate coordinates (multiple accessions per location)
- Silhouette coefficient evaluation for cluster quality
- Ensures environmental diversity across splits

**Output:**
- `splits_all.json` - Train/val/test accession IDs
  ```json
  {
      "train": ["1001", "1002", ...],
      "val": ["1234", "1235", ...],
      "test": ["2001", "2002", ...]
  }
  ```
- Distribution plots per variable: `plots_test/{variable}_distribution.png`

**Quality Metrics:**

After splitting, evaluate distribution similarity using:
- **Kolmogorov-Smirnov test:** Compare train/val/test distributions
- **Visual inspection:** Histogram overlays for each variable
- **Cluster representation:** Verify each cluster in all splits

---

### Step 6: Split Visualization

**Script:** `plot_splits.py`

Creates geographic and distribution visualizations to validate split quality.

**Geographic Visualization:**

**Projection:** North Polar Stereographic
- Focuses on Northern Hemisphere (primary *Arabidopsis* range)
- Shows spatial distribution of train/val/test samples

**Features:**
- Base layer: Country boundaries (gray)
- Sample points: Colored by split (blue/orange/green)
- Optional: Single color mode for overview (red)
- High resolution: 300 DPI for publication
- Map extent: Automatically adjusted to data range

**Color Scheme:**
- Train: Blue
- Validation: Orange
- Test: Green

**Output:**
- `plot_splits.png` - Geographic distribution map

**Optional Raster Overlay:**

Uncomment to overlay bioclimatic rasters (e.g., temperature):
```python
with rasterio.open("wc2.1_2.5m_bio_1.tif") as src:
    bio1 = src.read(1)
    ax.imshow(bio1, extent=(...), cmap="viridis", alpha=0.5)
```

---

### Step 7: Distribution Comparison

**Script:** `compare_variable_distributions.py`

Validates variable combination by comparing original and combined variable distributions.

**Input:**
- `coords_with_bioclim_30s_fixed.csv` - Original bioclimatic data
- `coords_with_soil.csv` - Original soil data
- `combined_variables_dataset_normalized.csv` - Combined dataset
- `variable_combination_mapping.csv` - Variable mapping

**Visualization:**

**A4-optimized multi-panel plots:**
- 2 columns × N rows layout
- Middle row variables highlighted (typically most informative)
- Each panel shows:
  - Original variables (normalized separately): Blue/green/orange histograms
  - Combined variable: Red step histogram
  - Zero mean line (black dashed)

**Key Comparisons:**

1. **Distribution shape:** Combined should resemble average of originals
2. **Variance:** Combined may have reduced variance (averaging effect)
3. **Outliers:** Combination smooths extreme values
4. **Modality:** Multi-modal originals may produce uni-modal combined

**Output:**
- `Plots/normalized_original_vs_combined_distributions.png`
- `variable_combination_summary.csv` - Statistical summary

**Quality Checks:**

- **Mean ~0:** Confirms normalization worked
- **Std ~1:** Z-score normalization preserved
- **Overlap:** Combined distribution should span similar range as originals
- **Smoothness:** Combined should be less noisy than individual variables

---

## Supporting Files

### consensusClustering.py

**Implementation of Consensus Clustering** (Monti et al. 2003)

**Algorithm Overview:**

1. **Resampling:** Subsample data H times (e.g., 100 iterations)
2. **Clustering:** Apply KMeans for each K in [L, K] range
3. **Consensus Matrix:** Build co-occurrence matrix
   - Entry M[i,j] = proportion of times samples i and j clustered together
4. **Final Clustering:** Apply hierarchical clustering to consensus matrix
5. **Selection:** Choose K with maximum delta in area under CDF

**Key Methods:**
- `fit(data)` - Builds consensus matrices for all K values
- `predict()` - Returns labels for best K
- `predict_data(data)` - Applies best K to new data

**Advantages:**
- Robust to initialization randomness
- Quantifies cluster stability
- Avoids over-fitting to single run

**Parameters:**
- `L` - Minimum number of clusters
- `K` - Maximum number of clusters
- `H` - Number of resampling iterations
- `resample_proportion` - Fraction to sample (default: 0.5)

---

## Configuration

### Common Parameters

**Correlation Analysis:**
- Correlation threshold: |r| > 0.9 for high correlation
- Clustering method: Ward linkage with optimal ordering
- Distance metric: √(2 × (1 - |r|))

**Normalization:**
- Method: Z-score (mean=0, std=1)
- Alternative methods: MinMax, Robust, None
- Applied before variable combination

**Data Splitting:**
- Test: 18% of samples
- Validation: 10% of samples
- Training: 72% of samples
- Consensus clustering: 100 iterations, K ∈ [25, 50]

---

## Dependencies

### Python Packages
- `numpy` - Numerical operations
- `pandas` - Data manipulation
- `matplotlib` - Static plotting
- `seaborn` - Statistical visualizations
- `scikit-learn` - Machine learning utilities
  - `StandardScaler` - Normalization
  - `KMeans` - Clustering
  - `AgglomerativeClustering` - Hierarchical clustering
- `scipy` - Scientific computing
  - `stats` - Statistical tests
  - `cluster.hierarchy` - Dendrogram utilities
- `geopandas` - Geographic data
- `cartopy` - Map projections
- `rasterio` - Raster data I/O
- `requests` / `urllib` - API access

### External Data Sources
- **WorldClim 2.1** - Bioclimatic variables
  - Resolution: 30 arc-seconds (~1 km²)
  - Variables: BIO1-BIO19
  - URL: https://www.worldclim.org/

- **SoilGrids** - Soil properties
  - Resolution: 250m
  - Coverage: Global
  - API: https://maps.isric.org/

- **1001 Genomes** - *Arabidopsis* genomic sequences
  - Pseudogenomes for 1,135 accessions
  - API: https://tools.1001genomes.org/

## Data Schema

### Environmental Variables CSV

**Columns:**
- `FID` - Family ID (integer)
- `IID` - Individual ID (integer or string)
- `LONG` - Longitude (decimal degrees)
- `LAT` - Latitude (decimal degrees)
- `BIO1` to `BIO19` - Bioclimatic variables (numeric)
- `clay_mean`, `bdod_mean`, etc. - Soil properties (numeric)
- `*_uncertainty` - Uncertainty estimates (numeric, optional)

**Missing Values:**
- Sentinel: `-3.4e+38`
- Action: Drop rows with missing values

### Combined Variables CSV

**Columns:**
- `FID` - Family ID
- `IID` - Individual ID
- `LONG`, `LAT` - Coordinates
- `BIO_*` - Combined bioclimatic variables (Z-score normalized)
- `Soil_*` - Combined soil variables (Z-score normalized)
- Individual variables not in clusters (Z-score normalized)

**Properties:**
- All numeric columns: Mean ≈ 0, Std ≈ 1
- No missing values
- Same sample set as input

### Splits JSON

**Structure:**
```json
{
    "train": [6909, 9728, 9888, ...],
    "val": [9433, 9626, 9755, ...],
    "test": [9559, 9618, 9647, ...]
}
```

**Properties:**
- No overlap between splits
- IDs correspond to `IID` column
- Stratified by environmental clustering

---

## Validation and Quality Control

### Data Quality Checks

**Pre-processing:**
1. ✓ Check for sentinel values (`-3.4e+38`)
2. ✓ Verify coordinate ranges (LONG: -180 to 180, LAT: -90 to 90)
3. ✓ Confirm sample IDs match across data sources
4. ✓ Validate SNP missingness and MAF thresholds

**Post-processing:**
1. ✓ Verify normalization (mean ≈ 0, std ≈ 1)
2. ✓ Check no overlap between train/val/test
3. ✓ Confirm proportional split sizes
4. ✓ Validate distribution similarity across splits (K-S test)
5. ✓ Check cluster representation in all splits

### Expected Outputs

**Sample counts (typical):**
- Total accessions: ~1,000-1,135
- Training: ~720
- Validation: ~100
- Test: ~180

**Variable reduction:**
- Original: ~32 environmental variables
- Combined: ~20-25 variables
- Reduction: ~20-35%

**Correlation clusters:**
- Number of clusters: 5-8 (|r| > 0.9)
- Cluster sizes: 2-3 variables each
- Total clustered: 10-20 variables

---
