# Attribution Analysis Pipeline

This directory contains scripts for explaining model predictions through attribution analysis, motif discovery, and enrichment analysis. The workflow progresses from model weight interpretation to identification of important genomic motifs.

## Pipeline Overview

The attribution analysis pipeline consists of the following major steps:

1. **Model Weight Explanation** - Extract and interpret important weights from trained models
2. **Saliency Map Generation** - Compute integrated gradients to attribute predictions to input sequences
3. **Gradient Peak Extraction** - Identify genomic positions with highest attribution scores
4. **Motif Enrichment Analysis** - Discover enriched sequence motifs using MEME Suite
5. **Motif Comparison & Analysis** - Compare discovered motifs against known databases

## Quick Start

```bash
# 1. Explain model weights
sbatch run_explain_model_weights.sh

# 2. Generate saliency maps (integrated gradients)
sbatch run_saliency_maps.sh

# 3. Extract gradient peak positions
sbatch run_extract_gradient_peaks.sh

# 4. Run motif enrichment analysis (MEME SEA)
sbatch memesuite_sea.sh

# 5. Filter significantly enriched motifs
sbatch filter_significant_motifs.sh

# 6. Run self-comparison to find redundant motifs
sbatch memesuite_tomtom.sh  # (with self-comparison settings)

# 7. Filter non-redundant motifs
sbatch filter_nonredundant_motifs.sh

# 8. Compare against known motif databases
sbatch memesuite_tomtom.sh  # (with JASPAR database)

# 9. Analyze Tomtom results
sbatch run_analyze_tomtom_results.sh
```

## Detailed Workflow

### Step 1: Model Weight Explanation

**Script:** `explain_model_weights.py`  
**SLURM:** `run_explain_model_weights.sh`

Extracts and interprets model weights to identify which SNP/embedding dimension pairs are most important for predictions.

**Input:**
- Trained model file (`.pkl`)
- Embedding dimension size (default: 256)

**Output:**
- `weights_explanation.csv` - Full weight explanation with SNP indices and embedding dimensions
- `weights_explanation_top_weights.csv` - Top N weights per variable

**Key Features:**
- Extracts linear model coefficients
- Maps weights to specific SNP indices and embedding dimensions
- Identifies top N most important features per target variable
- Optionally computes attribution scores (requires GPU and sequences)

**Usage:**
```bash
python explain_model_weights.py \
    --model model.pkl \
    --embedding_dim 256 \
    --output weights_explanation.csv \
    --top_n 10 \
    --explain_only  # Skip attribution analysis
```

---

### Step 2: Saliency Map Generation

**Script:** `generate_saliency_maps.py`  
**SLURM:** `run_saliency_maps.sh`

Computes integrated gradient saliency maps to identify which nucleotide positions in sequences contribute most to model predictions. Uses bidirectional analysis (forward + reverse complement) and multiple sequence alignment for robust attribution.

**Input:**
- Caduceus model path
- FASTA directory with per-accession sequences
- Validation accession IDs
- Weights file from Step 1 (or manual SNP:dimension pairs)

**Output (per variable subfolder):**
- `integrated_gradients_averaged_SNP{i}_dim{j}.csv` - Averaged saliency scores across sequences
  - Includes mean saliency, standard deviation, sample size
  - Nucleotide frequency percentages at each position
  - Gap percentages (from alignment)
- `raw_attributions_averaged_SNP{i}_dim{j}.csv` - Raw attribution vectors (optional)
- Visualization plots (if `--generate_plots` enabled)
- `alignments/` directory with multiple sequence alignments for each SNP

**Key Features:**
- Integrated gradients attribution (robust for Mamba/state-space models)
- Bidirectional processing (forward + reverse complement)
- Multiple sequence alignment to handle length variations
- Averaged results across all accessions for robust patterns
- Position-wise nucleotide frequency analysis
- Alignment caching for repeated runs

**Usage:**
```bash
python generate_saliency_maps.py \
    --caduceus_model_path "kuleshov-group/caduceus-ph_seqlen-131k_d_model-256_n_layer-16" \
    --fasta_dir /path/to/sequences \
    --val_ids val_ids.txt \
    --weights_file weights_explanation.csv \
    --top_n_weights 50 \
    --output_dir saliency_results \
    --generate_plots \
    --pooling max
```

**Special Options:**
- `--align-only` - Create alignments only without computing gradients
- `--target_variable` - Process only a specific target variable
- `--force` - Recompute existing files
- `--max_length` - Trim sequences to maximum length

---

### Step 3: Gradient Peak Extraction

**Script:** `extract_gradient_peaks.py`  
**SLURM:** `run_extract_gradient_peaks.sh`

Extracts gradient peak positions from saliency maps and converts them to genomic coordinates. Identifies multiple non-overlapping peaks per SNP/dimension using z-score thresholding.

**Input:**
- Integrated gradients CSV files from Step 2
- FASTA files with genomic coordinates in headers

**Output:**
- `gradient_peaks_summary.csv` - Summary of all identified peaks
  - SNP index, embedding dimension, peak index
  - Genomic coordinates (chromosome, absolute position)
  - Peak gradient value and z-score
  - Nucleotide frequency at peak position
- `gradient_peak_motifs.meme` - MEME format motif file with sequence windows around peaks

**Key Features:**
- Finds up to 3 non-overlapping peaks per SNP/dimension
- Z-score threshold (default: 1.96 for 95% confidence)
- Extracts genomic coordinates from FASTA headers
- Creates consensus sequence representations
- Generates motif representations for enrichment analysis

**Usage:**
```bash
python extract_gradient_peaks.py \
    --gradients_dir Results/Saliency/Variable \
    --fasta_dir Data/Pseudogenomes \
    --output_dir gradient_peaks \
    --window_size 11
```

---

### Step 4: Motif Enrichment Analysis (SEA)

**Script:** MEME Suite's `sea` command  
**SLURM:** `memesuite_sea.sh`

Tests for enrichment of discovered motifs against background sequences using MEME Suite's Simple Enrichment Analysis.

**Input:**
- Motif file from Step 3 (`gradient_peak_motifs.meme` or filtered versions)
- Background sequence FASTA file
- Optional: Primary sequence set for differential analysis

**Output (per seed):**
- `sea.tsv` - Enrichment results with p-values, q-values, enrichment ratios
- `sea.html` - HTML report with visualizations

**Key Features:**
- Multiple random seeds for robust results (array job)
- Enrichment ratio and score thresholds
- P-value and q-value (FDR) correction

**Usage:**
```bash
# Runs as SLURM array job (10 seeds by default)
sbatch memesuite_sea.sh
```

**Configuration in script:**
- `FASTA` - Background sequences
- `MOTIFS` - Query motifs
- `BASE_OUTDIR` - Output directory base
- `--seed` - Random seed (from array task ID)

---

### Step 5: Filter Significant Motifs

**Script:** `filter_significant_motifs.py`  
**SLURM:** `filter_significant_motifs.sh`

Filters motifs that are significantly enriched across multiple SEA runs using consensus thresholding.

**Input:**
- Original MEME motif file
- SEA results (multiple runs from Step 4)
- Enrichment ratio threshold (default: 2.5)
- Score threshold (default: 10)

**Output:**
- Filtered MEME file with significantly enriched motifs only

**Key Features:**
- Consensus filtering across multiple SEA runs
- Three aggregation modes: `any`, `majority`, `all`
- Configurable enrichment ratio and score thresholds
- Automatically discovers SEA results in subdirectories

**Usage:**
```bash
python filter_significant_motifs.py \
    --meme_file gradient_peak_motifs.meme \
    --sea_dir SEA_Results/ \
    --output_file significant_motifs.meme \
    --ratio 2.5 \
    --score_thresh 10 \
    --aggregate majority \
    --min_support 5
```

---

### Step 6: Motif Self-Comparison (Tomtom)

**Script:** MEME Suite's `tomtom` command  
**SLURM:** `memesuite_tomtom.sh` (configured for self-comparison)

Compares motifs against themselves to identify redundant/similar motifs.

**Input:**
- Motif file (e.g., significantly enriched motifs from Step 5)
- Same motif file as database

**Output:**
- `tomtom.tsv` - Pairwise similarity scores
- `tomtom.html` - HTML report with alignments

**Configuration:**
Set both query and database to the same motif file for self-comparison.

---

### Step 7: Filter Non-Redundant Motifs

**Script:** `filter_nonredundant_motifs.py`  
**SLURM:** `filter_nonredundant_motifs.sh`

Clusters redundant motifs and selects one representative per cluster based on Tomtom self-comparison results.

**Input:**
- Tomtom self-comparison TSV from Step 6
- Original MEME motif file

**Output:**
- Non-redundant MEME motif file

**Key Features:**
- Graph-based clustering using q-value threshold
- Selects representative with lowest sum of q-values to cluster members
- Preserves isolated (unique) motifs

**Usage:**
```bash
python filter_nonredundant_motifs.py \
    --tomtom_tsv tomtom_self.tsv \
    --meme_in motifs.meme \
    --meme_out motifs_nonredundant.meme \
    --qval_thresh 0.01
```

---

### Step 8: Compare Against Known Databases (Tomtom)

**Script:** MEME Suite's `tomtom` command  
**SLURM:** `memesuite_tomtom.sh` (configured for database comparison)

Compares discovered motifs against known transcription factor binding site databases (e.g., JASPAR).

**Input:**
- Non-redundant motif file from Step 7
- Known motif database (e.g., JASPAR plants)

**Output:**
- `tomtom.tsv` - Best matches to known motifs
- `tomtom.html` - HTML report with alignments

**Configuration:**
- `MOTIFS` - Query motifs (non-redundant)
- `DATABASE` - JASPAR or other motif database

---

### Step 9: Analyze Tomtom Results

**Script:** `analyze_tomtom_results.py`  
**SLURM:** `run_analyze_tomtom_results.sh`

Comprehensive analysis of Tomtom results including motif statistics and biological annotations.

**Input:**
- MEME motif file
- Tomtom TSV results
- Q-value threshold (default: 0.05)

**Output:**
- `tomtom_analysis_report.txt` - Comprehensive analysis report including:
  - Motif statistics (GC content, length, base composition)
  - Significant matches with JASPAR annotations
  - Non-significant matches
  - Novel motifs (no database matches)
  - Transcription factor class/family distribution

**Key Features:**
- Fetches JASPAR annotations via API
- Categorizes motifs by significance
- Statistical analysis of motif properties
- Counts query-to-target mappings

**Usage:**
```bash
python analyze_tomtom_results.py \
    --meme_file motifs.meme \
    --tomtom_tsv tomtom.tsv \
    --output_dir analysis_results \
    --qval_thresh 0.05
```

---

## File Naming Conventions

### Output Files
- `weights_explanation.csv` - Model weight interpretations
- `integrated_gradients_averaged_SNP{i}_dim{j}.csv` - Saliency maps
- `gradient_peaks_summary.csv` - Peak positions
- `gradient_peak_motifs.meme` - Discovered motifs
- `*_nonredundant.meme` - Filtered non-redundant motifs
- `*_enriched.meme` - Significantly enriched motifs
- `tomtom.tsv` - Motif comparison results
- `sea.tsv` - Enrichment analysis results

---

## Configuration

### Common Parameters

**Model & Data:**
- `--caduceus_model_path` - HuggingFace model ID or local path
- `--fasta_dir` - Directory with per-accession FASTA files
- `--val_ids` - File with validation accession IDs
- `--embedding_dim` - Model embedding dimension (default: 256)

**Analysis Options:**
- `--top_n` / `--top_n_weights` - Number of top features to analyze
- `--window_size` - Sequence window size around peaks
- `--qval_thresh` - Q-value threshold for significance
- `--force` - Force recomputation of existing results

**Computational:**
- `--pooling` - max or average pooling (default: max)
- `--attribution_steps` - Integration steps for gradients (default: 50)

### SLURM Resource Requirements

| Script | Partition | GPU | Memory | Time | Notes |
|--------|-----------|-----|--------|------|-------|
| explain_model_weights | gpu | 1 | 64G | 96h | Requires GPU for Caduceus |
| generate_saliency_maps | gpu | 1 | 128G | 96h | Most memory-intensive |
| extract_gradient_peaks | cpu | 0 | 16G | 2h | CPU only |
| memesuite_sea | cpu | 0 | 8G | 4h | Array job (10 tasks) |
| filter_significant_motifs | cpu | 0 | 4G | 20m | Fast filtering |
| memesuite_tomtom | cpu | 0 | 32G | 4h | Database size dependent |
| filter_nonredundant_motifs | cpu | 0 | 4G | 30m | Graph clustering |
| analyze_tomtom_results | cpu | 0 | 8G | 30m | API queries |

---

## Dependencies

### Python Packages
- `numpy` - Numerical operations
- `pandas` - Data manipulation
- `torch` - PyTorch for Caduceus model
- `transformers` - HuggingFace model loading
- `biopython` - FASTA parsing and sequence analysis
- `matplotlib`, `seaborn` - Visualization
- `tqdm` - Progress bars
- `scipy` - Statistical tests
- `statsmodels` - Multiple testing correction
- `networkx` - Graph operations for clustering
- `requests` - JASPAR API queries

### External Tools
- **MEME Suite** (>= 5.5.0)
  - `sea` - Simple Enrichment Analysis
  - `tomtom` - Motif comparison
- **MAFFT** or **ClustalW** - Multiple sequence alignment
- **CUDA** - GPU computation (for Steps 1-2)

### Singularity Containers
- `caduceus.sif` - Caduceus model and Python environment
- `caduceus_align.sif` - Caduceus with alignment tools
- `memesuite_latest.sif` - MEME Suite tools
- `memelite.sif` - Lightweight MEME environment

---

## Troubleshooting

### Common Issues

**1. "CUDA not available" error**
- Ensure SLURM job requests GPU: `#SBATCH --gres=gpu:1`
- Use `--nv` flag with Singularity
- Check GPU allocation: `nvidia-smi`

**2. "Alignment file not found"**
- Run with `--align-only` first to create alignments
- Check FASTA directory structure (per-accession files)
- Verify SNP indices match sequence indices in files

**3. "No gradients computed"**
- Check tokenizer compatibility with model
- Verify sequence length vs. max_length setting
- Ensure embedding dimension exists in model

**4. Memory issues**
- Reduce `--top_n_weights` for fewer SNP/dim pairs
- Use `--max_length` to trim long sequences
- Increase SLURM memory allocation
- Process variables separately with `--target_variable`

**5. Empty motif files**
- Check z-score threshold (peaks may be too weak)
- Verify gradient files have valid data
- Ensure FASTA headers contain coordinate information



