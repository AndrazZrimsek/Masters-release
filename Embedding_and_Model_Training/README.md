# Embedding and Model Training Pipeline

This directory contains scripts for generating DNA sequence embeddings using pretrained foundation models and training machine learning models to predict environmental variables from these embeddings. The pipeline transforms genomic sequences into dense vector representations and builds predictive models linking genotype to phenotype.

## Pipeline Overview

The embedding and model training workflow consists of two major phases:

1. **Sequence Embedding** - Transform DNA sequences into fixed-dimension vectors using Caduceus or other foundation models
2. **Predictive Modeling** - Train regression models to predict environmental variables from embeddings

## Quick Start

```bash
# 1. Generate embeddings from genomic sequences
sbatch run_embed_caduceus.sh

# 2. Train and evaluate prediction models
sbatch run_train_predict_models.sh
```

## Detailed Workflow

### Phase 1: Sequence Embedding

**Script:** `embed_sequences.py`  
**SLURM:** `run_embed_caduceus.sh`

Generates dense vector embeddings from DNA sequences using pretrained foundation models. Supports bidirectional processing (forward + reverse complement) and multiple pooling strategies.

**Supported Models:**

| Model | Description | Max Length | Embedding Dim |
|-------|-------------|------------|---------------|
| **Caduceus** | BiMamba architecture for DNA | 131k bp | 256 |
| **Nucleotide Transformer** | BERT-based DNA model | Variable | 768-1024 |
| **Enformer** | Transformer for genomic regulatory elements | 196k bp | 3072 |

**Input:**
- FASTA files per accession (`{accession_id}.fa`)
- Sequences organized by chromosome with headers:
  ```
  >{strain}|{chromosome}|{start}|{end}|{region}
  ATCGATCG...
  ```
- Accession IDs list (`columnIDs.txt`)

**Output:**
- Embeddings pickle file: `{name}_embeddings.pkl`
  ```python
  {
      'accession_id_1': [embedding_array_1, embedding_array_2, ...],
      'accession_id_2': [embedding_array_1, embedding_array_2, ...],
      ...
  }
  ```

**Key Features:**

**1. Bidirectional Embedding (RCPS - Reverse Complement Pair Symmetry)**
```python
# Forward sequence
embedding_fwd = model(sequence)

# Reverse complement
rc_sequence = reverse_complement(sequence)
embedding_rc = model(rc_sequence)

# Average both directions
embedding = (embedding_fwd + embedding_rc.flip(dims=[1])) / 2.0
```

**2. Pooling Strategies:**
- `max`: Max pooling across sequence positions
  ```python
  embedding = torch.max(embedding, dim=1)[0]
  ```
- `avg`: Average pooling across sequence positions
  ```python
  embedding = torch.mean(embedding, dim=1)
  ```

**3. SNP Context Window (Optional):**
Centers pooling on a specific region around the sequence midpoint:
```python
center = seq_len // 2
start = center - window_size // 2
end = center + window_size // 2
embedding = pool(embedding[:, start:end])
```

**4. Region Joining Mode (`--join_regions`):**
Concatenates all genomic regions per accession with `[SEP]` tokens:
```python
# Join all sequences for accession
full_sequence = "[SEP]".join(all_sequences)

# Split into chunks at [SEP] boundaries (respecting max_length)
chunks = split_at_sep_tokens(full_sequence, max_tokens)

# Process each chunk
for chunk in chunks:
    embedding = model(chunk)
```

**Usage:**
```bash
python embed_sequences.py \
    --model_name_or_path "kuleshov-group/caduceus-ph_seqlen-131k_d_model-256_n_layer-16" \
    --accessions_dir /path/to/fasta/files \
    --output_file /path/to/output.pkl \
    --seq_len 131072 \
    --bp_per_token 1 \
    --pool_method max \
    --rcps \
    --snp_context_window 100
```

**Options:**
- `--model_name_or_path` - HuggingFace model ID or local path
- `--seq_len` - Maximum sequence length in base pairs (default: 131072)
- `--bp_per_token` - Base pairs per token (default: 1)
- `--pool_method` - Pooling strategy: `max` or `avg` (default: `max`)
- `--rcps` / `--no-rcps` - Enable/disable reverse complement averaging
- `--snp_context_window` - Window size around sequence center for pooling
- `--join_regions` - Concatenate all sequences per accession with `[SEP]`
- `--accessions_dir` - Directory containing FASTA files
- `--output_file` - Output pickle file path
- `--name` - Embedding model name for output file

**Distributed Training:**

The script uses PyTorch DistributedDataParallel (DDP):
```bash
export RANK=0
export WORLD_SIZE=1
export MASTER_ADDR=localhost
export MASTER_PORT=29321

srun singularity exec --nv containers/caduceus.sif \
    python embed_sequences.py [args...]
```

**Model-Specific Handling:**

**Caduceus:**
- Loaded via `AutoModel.from_pretrained()`
- Returns `last_hidden_state` directly
- Supports long sequences (up to 131k bp)

**Nucleotide Transformer:**
- Loaded via `AutoModelForMaskedLM`
- Backbone accessed via `.esm` attribute
- Requires `trust_remote_code=True`

**Enformer:**
- Loaded via `enformer_pytorch.from_pretrained()`
- Custom tokenizer (A→0, C→1, G→2, T→3, N→4)
- Returns embeddings with `return_embeddings=True`

**Memory Optimization:**
- Mixed precision training (FP16): `torch.autocast(device_type="cuda", dtype=torch.float16)`
- Batch processing with `embed_dump_batch_size`
- Gradient computation disabled: `torch.no_grad()`

**Reverse Complement Implementation:**
```python
STRING_COMPLEMENT_MAP = {
    "A": "T", "C": "G", "G": "C", "T": "A",
    "a": "t", "c": "g", "g": "c", "t": "a",
    "N": "N", "n": "n"
}

def string_reverse_complement(seq, sep_token='[SEP]'):
    # Split by [SEP] tokens
    segments = seq.split(sep_token)
    
    # Reverse complement each segment
    rc_segments = []
    for segment in segments:
        rc = ''.join(STRING_COMPLEMENT_MAP.get(base, base) 
                     for base in segment[::-1])
        rc_segments.append(rc)
    
    # Reverse segment order and rejoin
    return sep_token.join(rc_segments[::-1])
```

**Output Structure:**

Embeddings are saved as a nested dictionary:
```python
{
    'accession_6909': [
        array([0.12, -0.45, ..., 0.89]),  # Embedding for region 1
        array([0.34, -0.23, ..., 0.56]),  # Embedding for region 2
        ...
    ],
    'accession_9728': [...],
    ...
}
```

**Embedding Dimensions:**
- Without pooling: `(n_regions, seq_len // bp_per_token, embedding_dim)`
- With pooling: `(n_regions, embedding_dim)`
- With `--join_regions`: `(n_chunks, embedding_dim)`

---

### Phase 2: Predictive Modeling

**Script:** `train_predict_models.py`  
**SLURM:** `run_train_predict_models.sh`

Trains regression models to predict environmental variables from genomic embeddings. Supports multiple model types with hyperparameter tuning via GridSearchCV.

**Input:**
- Embeddings pickle files from Phase 1
- Environmental variables CSV (normalized)
- Train/validation ID files
- Model configuration parameters

**Output:**
- Trained models: `{model_type}_model_{embedding_name}.pkl`
- Results CSV: `multioutput_results_{job_id}.csv`
- Per-variable summary: `results_{embedding_name}.txt`

**Supported Models:**

**1. ElasticNet (Default)**
- Linear regression with L1 + L2 regularization
- Fast training, interpretable weights
- Handles high-dimensional embeddings well

**Hyperparameters:**
```python
param_grid = {
    'alpha': np.logspace(-2, 1, 16),  # Regularization strength
    'l1_ratio': [0.1],                # L1/L2 mix (0.1 = more L2)
    'max_iter': [25000]               # Convergence iterations
}
```

**2. Support Vector Regression (SVR)**
- Non-linear regression with kernel trick
- RBF or linear kernel options
- Slower but potentially higher accuracy

**Hyperparameters:**
```python
param_grid = {
    'C': np.logspace(-6, 2, 10),      # Regularization parameter
    'gamma': ['scale'],               # Kernel coefficient
    'kernel': ['linear', 'rbf']       # Kernel type
}
```

**Data Processing Pipeline:**

**1. Embedding Loading:**
```python
# Load embeddings from pickle files
embeddings = load_embeddings(embeddings_dir)

# Structure: {embedding_file: {accession_id: embedding_array}}
```

**2. Embedding Aggregation Strategies:**

**Strategy A: Horizontal Stacking (Default)**
```python
# Stack all region embeddings side-by-side
embedding = np.hstack([emb_region1, emb_region2, ...])
# Shape: (n_regions * embedding_dim,)
```

**Strategy B: Vertical Stacking (`vstack=True`)**
```python
# Stack region embeddings as rows
embedding = np.vstack([emb_region1, emb_region2, ...])
# Shape: (n_regions, embedding_dim)
```

**Strategy C: Averaging (`--average-embeddings`)**
```python
# Average all region embeddings to single vector
embedding = np.mean([emb_region1, emb_region2, ...], axis=0)
# Shape: (embedding_dim,)
```

**3. Optional Dimensionality Reduction (`--svd-dim`):**
```python
from sklearn.decomposition import TruncatedSVD

# Reduce embedding columns via SVD
svd = TruncatedSVD(n_components=svd_dim, random_state=42)
reduced_embedding = svd.fit_transform(embedding)
# Shape: (n_snps, svd_dim) → flattened to (n_snps * svd_dim,)
```

**4. Data Alignment:**
```python
# Match embeddings with environmental data
common_ids = set(embedding_ids) ∩ set(env_ids)

# Filter to train/validation splits
train_ids = [id for id in train_ids if id in common_ids]
val_ids = [id for id in val_ids if id in common_ids]

# Create matrices
X_train = np.array([embeddings[id] for id in train_ids])
y_train = env_data.loc[train_ids, env_vars].values
```

**5. Standardization:**
```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
```

**Training Procedure:**

**Multi-Output Strategy:**

Each environmental variable is predicted independently with per-variable hyperparameter tuning:

```python
for i, var_name in enumerate(env_vars):
    # Extract single target variable
    y_train_var = y_train[:, i]
    y_val_var = y_val[:, i]
    
    # Hyperparameter tuning via GridSearchCV
    grid_search = GridSearchCV(
        estimator=ElasticNet(),
        param_grid=param_grid,
        cv=5,                          # 5-fold cross-validation
        scoring='neg_mean_squared_error',
        n_jobs=-1                      # Parallel processing
    )
    
    # Fit and select best model
    grid_search.fit(X_train_scaled, y_train_var)
    best_model = grid_search.best_estimator_
    
    # Predict on train and validation sets
    y_train_pred[:, i] = best_model.predict(X_train_scaled)
    y_val_pred[:, i] = best_model.predict(X_val_scaled)
```

**Evaluation Metrics:**

**Per-Variable Metrics:**
- **R² Score:** Coefficient of determination (1.0 = perfect fit)
  ```python
  r2 = r2_score(y_true, y_pred)
  ```
- **MSE:** Mean squared error
  ```python
  mse = mean_squared_error(y_true, y_pred)
  ```
- **MAE:** Mean absolute error
  ```python
  mae = mean_absolute_error(y_true, y_pred)
  ```
- **Spearman Correlation:** Rank-order correlation (robust to outliers)
  ```python
  spearman = spearmanr(y_true, y_pred).correlation
  ```

**Overall Metrics:**
- Computed across all variables simultaneously
- Provides global model performance assessment

**Usage:**
```bash
python train_predict_models.py \
    --embeddings_dir /path/to/embeddings \
    --csv_files /path/to/environmental_data.csv \
    --train_ids /path/to/train_ids.txt \
    --val_ids /path/to/val_ids.txt \
    --output_dir /path/to/results \
    --model elasticnet \
    --job_id $SLURM_JOB_ID \
    --jobname $SLURM_JOB_NAME
```

**Options:**
- `--embeddings_dir` - Directory containing embedding pickle files
- `--csv_files` - List of CSV files with environmental data
- `--train_ids` - File with training accession IDs
- `--val_ids` - File with validation accession IDs
- `--output_dir` - Directory for saving results
- `--model` - Model type: `elasticnet` or `svm`
- `--job_id` - SLURM job ID for unique output filenames
- `--jobname` - Job name for output folder organization
- `--svd-dim` - Apply SVD to reduce embedding dimensions (e.g., 10)
- `--average-embeddings` - Average all region embeddings to single vector

**Output Files:**

**1. Trained Models:**
```python
# Saved as pickle file per embedding file
{
    'model': [best_model_var1, best_model_var2, ...],
    'scaler': StandardScaler(),
    'best_params': [{params_var1}, {params_var2}, ...],
    'env_vars': ['BIO1', 'BIO_11_6', ...],
    'train_ids': ['6909', '9728', ...],
    'val_ids': ['9433', '9626', ...],
    'individual_metrics': {
        'BIO1': {'train_r2': 0.85, 'val_r2': 0.78, ...},
        'BIO_11_6': {...},
        ...
    },
    'overall_metrics': {
        'overall_train_r2': 0.82,
        'overall_val_r2': 0.75,
        ...
    }
}
```

**2. Results CSV:**
```csv
data_combination,variable,model_type,train_r2,val_r2,train_mse,val_mse,train_spearman,val_spearman,best_params
Caduceus_512,BIO1,multioutput,0.852,0.783,0.148,0.217,0.891,0.824,{'alpha': 0.1}
Caduceus_512,BIO_11_6,multioutput,0.768,0.701,0.232,0.299,0.812,0.756,{'alpha': 0.05}
...
Caduceus_512,overall,multioutput,0.795,0.720,0.205,0.278,0.845,0.782,-
```

**3. Per-Variable Summary:**
```
BIO_ID R2_Coefficient Spearman_Coefficient
BIO1 0.783 0.824
BIO_11_6 0.701 0.756
BIO_10_5 0.689 0.743
clay_mean 0.534 0.612
...
```

**Hyperparameter Tuning Details:**

**ElasticNet Grid:**
| Parameter | Values | Purpose |
|-----------|--------|---------|
| `alpha` | 0.01, 0.02, ..., 10.0 (16 values, log-spaced) | Regularization strength |
| `l1_ratio` | 0.1 | L1/L2 penalty mix (0.1 = 90% L2, 10% L1) |
| `max_iter` | 25000 | Maximum iterations for convergence |

**SVR Grid (Linear Kernel):**
| Parameter | Values | Purpose |
|-----------|--------|---------|
| `C` | 1e-6, 1e-5, ..., 100 (10 values, log-spaced) | Regularization parameter |
| `gamma` | 'scale' | Kernel coefficient |
| `kernel` | 'linear' | Linear kernel for interpretability |

**SVR Grid (RBF Kernel):**
| Parameter | Values | Purpose |
|-----------|--------|---------|
| `C` | 10, 20, ..., 100 (6 values, log-spaced) | Regularization parameter |
| `gamma` | 'scale', 'auto', 0.01, 0.1, 1 | Kernel coefficient |
| `kernel` | 'rbf' | Radial basis function kernel |

**Cross-Validation Strategy:**
- 5-fold cross-validation within training set
- Scoring: Negative MSE (lower is better)
- Parallel processing: All CPU cores (`n_jobs=-1`)

---

## SLURM Scripts

### run_embed_caduceus.sh

**Resource Requirements:**
```bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=92:00:00
```

**Environment Setup:**
```bash
export RANK=0
export WORLD_SIZE=1
export MASTER_ADDR=localhost
export MASTER_PORT=29321
```

**Execution:**
```bash
srun singularity exec --nv containers/caduceus.sif bash -c "
    python ./embed_sequences.py \
        --model_name_or_path kuleshov-group/caduceus-ph_seqlen-131k_d_model-256_n_layer-16 \
        --accessions_dir /path/to/fasta \
        --output_file /path/to/output.pkl \
        --pool_method max
"
```

**Container:** `caduceus.sif` with CUDA support (`--nv`)

---

### run_train_predict_models.sh

**Resource Requirements:**
```bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=48:00:00
```

**Execution:**
```bash
srun singularity exec containers/container-pt.sif \
    python train_predict_models.py \
        --embeddings_dir /path/to/embeddings \
        --csv_files /path/to/environmental_data.csv \
        --train_ids /path/to/train_ids.txt \
        --val_ids /path/to/val_ids.txt \
        --output_dir /path/to/results \
        --model elasticnet \
        --job_id "$SLURM_JOB_ID" \
        --jobname "$SLURM_JOB_NAME"
```

**Container:** `container-pt.sif` with scikit-learn, pandas, numpy

---

## File Naming Conventions

### Input Files
- `{accession_id}.fa` - FASTA file per accession
- `columnIDs.txt` - List of accession IDs (tab or newline separated)
- `combined_variables_dataset_normalized.csv` - Environmental variables
- `train_ids.txt`, `val_ids.txt` - Train/validation splits

### Output Files
- `{name}_embeddings.pkl` - Embeddings dictionary
- `{model_type}_model_{embedding_name}.pkl` - Trained model
- `multioutput_results_{job_id}.csv` - Full results table
- `results_{embedding_name}.txt` - Per-variable summary

---

## Configuration

### Embedding Parameters

| Parameter | Default | Range | Purpose |
|-----------|---------|-------|---------|
| `seq_len` | 131072 | 512-196608 | Max sequence length (bp) |
| `bp_per_token` | 1 | 1-8 | Bases per token |
| `pool_method` | `max` | `max`, `avg` | Pooling strategy |
| `snp_context_window` | `None` | 10-1000 | Context window size |
| `rcps` | `False` | Boolean | Use reverse complement |
| `join_regions` | `False` | Boolean | Concatenate sequences |

### Model Training Parameters

| Parameter | Default | Range | Purpose |
|-----------|---------|-------|---------|
| `model` | `elasticnet` | - | Model type |
| `alpha` (ElasticNet) | Auto-tuned | 0.01-10 | Regularization |
| `l1_ratio` | 0.1 | 0.0-1.0 | L1/L2 mix |
| `C` (SVR) | Auto-tuned | 1e-6-100 | Regularization |
| `kernel` (SVR) | `linear` | `linear`, `rbf` | Kernel type |
| `cv` | 5 | 3-10 | CV folds |
| `svd_dim` | 0 | 0-1000 | SVD dimensions |

---

## Dependencies

### Python Packages

**Embedding Generation:**
- `torch` (≥1.13) - PyTorch for model inference
- `transformers` (≥4.30) - HuggingFace models
- `enformer-pytorch` - Enformer model (optional)
- `biopython` - FASTA file parsing
- `numpy` - Numerical operations
- `tqdm` - Progress bars

**Model Training:**
- `scikit-learn` (≥1.2) - Machine learning models
- `pandas` - Data manipulation
- `numpy` - Numerical operations
- `scipy` - Statistical functions

### External Resources

**Pretrained Models (HuggingFace):**
- `kuleshov-group/caduceus-ph_seqlen-131k_d_model-256_n_layer-16`
- `InstaDeepAI/nucleotide-transformer-v2-500m-multi-species`
- `EleutherAI/enformer-official-rough` (via enformer-pytorch)

**Singularity Containers:**
- `caduceus.sif` - Caduceus model + PyTorch + CUDA
- `container-pt.sif` - PyTorch + scikit-learn + pandas

### Installation

```bash
# Core packages
pip install torch transformers biopython numpy pandas scikit-learn scipy tqdm

# Optional: Enformer
pip install enformer-pytorch

# For GPU support
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

---

## Computational Requirements

### GPU Memory Usage (Embedding Generation)

| Model | Sequence Length | Batch Size | GPU Memory |
|-------|-----------------|------------|------------|
| Caduceus-256 | 131k bp | 1 | 40-60 GB |
| Caduceus-256 | 32k bp | 4 | 40-60 GB |
| NT-500M | 6k tokens | 1 | 24-32 GB |
| Enformer | 196k bp | 1 | 80-100 GB |

**Optimization Tips:**
- Use mixed precision (FP16): Reduces memory by ~50%
- Reduce batch size if OOM errors occur
- Use gradient checkpointing for very long sequences
- Process accessions sequentially rather than batching

### CPU Memory Usage (Model Training)

| Embedding Dim | Num Accessions | Num Regions | Memory |
|---------------|----------------|-------------|--------|
| 256 | 1000 | 100 | 8-16 GB |
| 256 | 1000 | 200 | 16-32 GB |
| 768 | 1000 | 100 | 24-48 GB |
| 768 | 1000 | 200 | 48-96 GB |

**Optimization Tips:**
- Use `--average-embeddings` to reduce memory
- Apply `--svd-dim` to compress embeddings
- Process embedding files sequentially
- Use vertical stacking for very large embeddings

### Runtime Estimates

**Embedding Generation (Caduceus, 1 GPU):**
- 100 accessions × 100 regions × 1kb sequences: ~2-4 hours
- 1000 accessions × 100 regions × 1kb sequences: ~20-40 hours
- 1000 accessions × 100 regions × 100kb sequences: ~60-90 hours

**Model Training (ElasticNet, 16 CPUs):**
- 1000 samples × 25600 features × 25 variables: ~2-6 hours
- 1000 samples × 51200 features × 25 variables: ~4-12 hours
- Grid search increases runtime by factor of (grid_size × cv_folds)

---

## Workflow Examples

### Example 1: Basic Embedding and Training

```bash
# Step 1: Generate embeddings
sbatch run_embed_caduceus.sh

# Wait for job completion...

# Step 2: Train models
sbatch run_train_predict_models.sh
```

### Example 2: Custom Embedding Configuration

```bash
# Generate embeddings with specific settings
python embed_sequences.py \
    --model_name_or_path kuleshov-group/caduceus-ph_seqlen-131k_d_model-256_n_layer-16 \
    --accessions_dir /data/pseudogenomes/SparSNP_512 \
    --output_file /results/embeddings_512bp_max.pkl \
    --seq_len 131072 \
    --pool_method max \
    --rcps \
    --snp_context_window 100
```

### Example 3: Training with Dimensionality Reduction

```bash
# Train with SVD-compressed embeddings
python train_predict_models.py \
    --embeddings_dir /results/embeddings \
    --csv_files /data/environmental_data.csv \
    --train_ids /data/train_ids.txt \
    --val_ids /data/val_ids.txt \
    --output_dir /results/models \
    --model elasticnet \
    --svd-dim 100
```

### Example 4: Multiple Model Comparison

```bash
# Train ElasticNet
python train_predict_models.py \
    --model elasticnet \
    --output_dir /results/elasticnet

# Train SVR (Linear)
python train_predict_models.py \
    --model svm \
    --output_dir /results/svr_linear

# Compare results
python -c "
import pandas as pd
enet = pd.read_csv('/results/elasticnet/multioutput_results.csv')
svr = pd.read_csv('/results/svr_linear/multioutput_results.csv')
print('ElasticNet mean R²:', enet[enet['variable']!='overall']['val_r2'].mean())
print('SVR mean R²:', svr[svr['variable']!='overall']['val_r2'].mean())
"
```

---

## Troubleshooting

### Common Issues

**Issue: CUDA Out of Memory**
```
RuntimeError: CUDA out of memory
```
**Solutions:**
- Reduce `seq_len` parameter
- Use mixed precision (already enabled by default)
- Process fewer regions per accession
- Use CPU fallback (slower): Remove `--gres=gpu:1`

**Issue: Tokenization Length Mismatch**
```
Warning: input_ids and rc_input_ids have different lengths
```
**Solutions:**
- Ensure sequences don't exceed `seq_len`
- Check for special characters in FASTA files
- Verify `[SEP]` token handling in `--join_regions` mode

**Issue: No Common IDs**
```
Warning: No common IDs found
```
**Solutions:**
- Check accession ID format consistency (string vs. int)
- Verify FASTA filenames match accession IDs
- Ensure environmental data has matching IDs
- Check for leading/trailing whitespace in ID files

**Issue: ElasticNet Convergence Warning**
```
ConvergenceWarning: Objective did not converge
```
**Solutions:**
- Increase `max_iter` in param_grid (e.g., 50000)
- Scale features if not already done (should be automatic)
- Check for zero-variance features
- Consider reducing `alpha` range

**Issue: SVD Dimension Error**
```
ValueError: n_components must be <= min(n_samples, n_features)
```
**Solutions:**
- Reduce `--svd-dim` value
- Ensure sufficient samples (rows) in embedding matrix
- Check embedding shape before SVD application

---

## Performance Metrics Interpretation

### R² Score (Coefficient of Determination)

| R² Range | Interpretation | Typical For |
|----------|----------------|-------------|
| 0.9-1.0 | Excellent fit | Temperature variables (BIO1, BIO5-11) |
| 0.7-0.9 | Good fit | Precipitation patterns (BIO12-19) |
| 0.5-0.7 | Moderate fit | Soil properties (clay, pH, SOC) |
| 0.3-0.5 | Weak fit | Highly variable traits |
| < 0.3 | Poor fit | Noise-dominated or complex traits |

### Spearman Correlation

| Spearman Range | Interpretation | Notes |
|----------------|----------------|-------|
| 0.8-1.0 | Strong monotonic relationship | Robust ranking preserved |
| 0.6-0.8 | Moderate relationship | Useful for rank-based analysis |
| 0.4-0.6 | Weak relationship | Better than R² for outliers |
| < 0.4 | Very weak | Model struggles with rank order |

**When to prefer Spearman over R²:**
- Presence of outliers in environmental data
- Non-linear relationships (Spearman captures monotonic trends)
- Rank-based predictions more important than absolute values
- Comparing models with different scales

---

## Best Practices

### Embedding Generation

1. **Always use RCPS** (`--rcps`) for DNA sequences to capture bidirectional information
2. **Choose pooling method** based on downstream task:
   - `max`: Better for identifying peak signals (e.g., regulatory elements)
   - `avg`: Better for global sequence representation
3. **Use SNP context window** when focusing on specific variant positions
4. **Monitor GPU memory** - reduce batch size if hitting OOM errors
5. **Validate embeddings** - check for NaN values and consistent dimensions

### Model Training

1. **Always standardize features** (done automatically in script)
2. **Use ElasticNet for interpretability** - coefficients show feature importance
3. **Use SVR for highest accuracy** - but slower and less interpretable
4. **Monitor convergence** - increase `max_iter` if warnings appear
5. **Cross-validate thoroughly** - 5-fold CV is minimum, consider 10-fold
6. **Check for overfitting** - compare train vs. validation metrics
7. **Save best models** - include scaler and hyperparameters for reproducibility

### Data Quality

1. **Remove low-quality sequences** before embedding
2. **Handle missing environmental data** - drop or impute
3. **Balance train/val splits** - ensure representative sampling
4. **Verify ID consistency** across all data sources
5. **Normalize environmental variables** before modeling

---

## Citation and References

### Foundation Models

**Caduceus (BiMamba for DNA):**
```
Schiff et al., 2024. Caduceus: Bi-Directional Equivariant Long-Range DNA 
Sequence Modeling. arXiv:2403.03234.
```

**Nucleotide Transformer:**
```
Dalla-Torre et al., 2023. The Nucleotide Transformer: Building and Evaluating 
Robust Foundation Models for Human Genomics. bioRxiv.
```

**Enformer:**
```
Avsec et al., 2021. Effective gene expression prediction from sequence by 
integrating long-range interactions. Nature Methods 18, 1196-1203.
```

### Methods

**ElasticNet Regression:**
```
Zou & Hastie, 2005. Regularization and variable selection via the elastic net. 
Journal of the Royal Statistical Society: Series B 67(2), 301-320.
```

**Support Vector Regression:**
```
Drucker et al., 1997. Support vector regression machines. 
Advances in Neural Information Processing Systems 9, 155-161.
```

---

## Future Enhancements

### Planned Features
- [ ] Multi-task learning across all variables simultaneously
- [ ] Attention-based pooling for sequence representations
- [ ] Integration of non-genetic features (climate, soil) as covariates
- [ ] Ensemble modeling (combine ElasticNet + SVR predictions)
- [ ] Uncertainty quantification (prediction intervals)

### Experimental Features
- [ ] Fine-tuning foundation models on environmental prediction task
- [ ] Graph neural networks for capturing SNP interactions
- [ ] Transformer-based regression heads
- [ ] Active learning for efficient data collection
- [ ] Transfer learning across species

---

## Contact and Support

For questions or issues with the embedding and model training pipeline:
- Check model documentation on HuggingFace
- Verify GPU/memory requirements
- Review troubleshooting section above
- Ensure container compatibility with SLURM environment

**Key Assumptions:**
- Pretrained models accessible via HuggingFace
- FASTA files formatted correctly with chromosome information
- Environmental variables are pre-normalized
- Train/validation splits are predefined
- Sufficient computational resources (GPU for embedding, CPU for training)
