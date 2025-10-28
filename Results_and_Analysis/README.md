# Results and Analysis

This directory contains Jupyter notebooks for analyzing and visualizing model performance results, along with pre-computed performance metrics comparing different model configurations and sequence length experiments.

## Contents

### Jupyter Notebooks

1. **`thesis_plotting.ipynb`** - Performance metric analysis and visualization
2. **`visualize_tsne.ipynb`** - Dimensionality reduction and embedding visualization

### Results Directory

Pre-computed performance metrics organized by experiment type:

- **Model comparison results** - ElasticNet, SVR Linear, SVR RBF performance
- **Sequence length experiments** - Performance across different sequence lengths
- **Size test experiments** - Dataset size impact on model performance
- **SparSNP comparison** - SNP selection strategy comparison

---

## Jupyter Notebooks

### 1. thesis_plotting.ipynb

**Purpose:** Analyze and visualize model performance metrics with statistical rigor.

**Key Features:**
- Load performance results from CSV files
- Compute mean R² and Spearman correlation across variables
- Bootstrap confidence intervals for performance metrics
- Statistical significance testing
- Publication-quality plots


### 2. visualize_embeddings.ipynb

**Purpose:** Visualize high-dimensional DNA embeddings using dimensionality reduction techniques.

**Key Features:**
- Load pretrained Caduceus embeddings (flattened 100×256 → 25,600 dimensions)
- PCA preprocessing for computational efficiency
- t-SNE, UMAP, and MDS dimensionality reduction
- Interactive Plotly visualizations
- Static matplotlib plots
- Metadata integration (Köppen-Geiger climate zones, bioclimatic variables)
- Clustering analysis and visualization
- Trustworthiness metric computation

**Workflow:**

1. **Load embeddings and metadata:**
   ```python
   EMBED_PKL = './Embeddings/fixedlen_25000/Caduceus_FixedLen_25000_avg.pkl'
   DATA_DRIVEN_CSV = './Data/coords_with_data_driven.csv'
   KOPPEN_CSV = './Data/coords_with_koppen_geiger.csv'
   BIOCLIM_CSV = './Data/coords_with_bioclim.csv'
   ```

2. **Reshape embeddings:**
   - Each sample: (100 SNPs, 256 embedding dimensions) → flatten to 25,600D vector
   - Stack all samples into array: (n_samples, 25600)

3. **PCA preprocessing:**
   - Reduce dimensionality before t-SNE/UMAP
   - Retain variance while improving computational efficiency
   - Typical: 50-100 principal components

4. **Dimensionality reduction:**
   - **t-SNE**: Non-linear, preserves local structure
   - **UMAP**: Faster, preserves both local and global structure
   - **MDS**: Classical multidimensional scaling

5. **Metadata integration:**
   - Join embeddings with climate zone classifications
   - Color points by Köppen-Geiger zones
   - Overlay bioclimatic variables (BIO1-BIO19)
   - Show clustering results (KM12, KM30, ISO16, RF30)

6. **Visualization:**
   - **Interactive Plotly plots** - Hover tooltips, zoom, pan
   - **Static matplotlib plots** - Publication-ready figures
   - **3D embeddings** - Optional third dimension
   - **Cluster overlays** - Visualize groupings

**Key Metadata Fields:**
- `IID` - Individual/accession identifier
- `Koppen-Geiger-Name` - Climate zone classification
- `KM12_value`, `KM30_value` - K-means clustering labels
- `ISO16_value` - Isodata clustering label
- `RF30_value` - Random forest clustering label
- `BIO1-BIO19` - Bioclimatic variables

**Quality Metrics:**
- **Trustworthiness** - Measures preservation of local neighborhoods
- **Stress** - MDS quality metric (lower is better)
- **Variance explained** - PCA component importance

**Output:**
- Saved plots (PNG, HTML, SVG)
- Embedding coordinates CSV
- Quality metrics report
- Interactive HTML dashboards

---

## Dependencies

### Python Packages
- `numpy` - Numerical operations
- `pandas` - Data manipulation
- `matplotlib` - Static plotting
- `plotly` - Interactive visualizations
- `scikit-learn` - PCA, t-SNE, metrics
- `umap-learn` - UMAP algorithm (optional)
- `scipy` - Statistical tests
- `jupyter` - Notebook environment

### Installation
```bash
pip install numpy pandas matplotlib plotly scikit-learn scipy jupyter
pip install umap-learn  # Optional, for UMAP
```
