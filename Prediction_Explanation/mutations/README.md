# Mutation Analysis Pipeline

This directory contains scripts for performing in silico mutation experiments to understand how sequence variants affect model predictions. The workflow generates mutated embeddings, analyzes their effects on predictions, and compares sequence motifs between high and low phenotype groups.

## Pipeline Overview

The mutation analysis pipeline consists of three major steps:

1. **Motif Comparison (Top/Bottom)** - Identify sequence differences between phenotypic extremes
2. **Mutation Embedding Generation** - Create embeddings for mutated sequences at important positions
3. **Mutation Effect Analysis** - Quantify prediction changes from mutations

## Quick Start

```bash
# 1. Compare motifs between top and bottom phenotype accessions
sbatch run_extract_top_bottom_motifs.sh

# 2. Generate mutation embeddings for selected positions
sbatch run_embed_caduceus_mutations.sh

# 3. Analyze mutation effects on predictions
sbatch run_analyze_mutation_effects.sh
```

## Detailed Workflow

### Step 1: Motif Comparison (Top vs Bottom Accessions)

**Script:** `extract_top_bottom_motifs.py`  
**SLURM:** `run_extract_top_bottom_motifs.sh`

Extracts accessions with extreme phenotype values and important genomic positions. Also does quick comparison of the Top vs. Bottom motif PFMs.

**Input:**
- Target variable name
- Phenotype CSV with accession IDs and trait values
- Validation accession IDs file
- Weights explanation CSV (from attribution analysis)
- Gradient peaks summary CSV (from attribution analysis)
- Alignment directory with SNP alignments

**Output (per weight or peak):**
- `selected_weight.json` - Weight metadata (SNP index, embedding dimension, weight value)
- `selected_peak.json` - Peak metadata (position, z-score)
- `top_group_motifs.fasta` - Sequence windows for top accessions
- `bottom_group_motifs.fasta` - Sequence windows for bottom accessions
- `top_group_pfm.tsv` - Position frequency matrix for top group
- `bottom_group_pfm.tsv` - Position frequency matrix for bottom group
- `motif_comparison.tsv` - Per-position comparison (deltas between groups)
- `consensus_top.txt` - Consensus sequence for top group
- `consensus_bottom.txt` - Consensus sequence for bottom group
- `summary.txt` - Analysis summary statistics
- `top_accessions.txt` - List of top phenotype accessions
- `bottom_accessions.txt` - List of bottom phenotype accessions

**Multi-weight/peak mode:**
When `--num-weights > 1` or `--peak-mode all`, creates subdirectories:
- `w{rank}_snp{snp}_dim{dim}/` - Per-weight subdirectories
- `peak{index}_*` - Per-peak files (when `--peak-mode all`)
- `selected_weights.csv` - Aggregated weight information
- `relevant_peaks.csv` - Aggregated peak information

**Key Features:**
- Stratifies accessions by phenotype values (top N vs bottom N)
- Extracts sequence windows around gradient peaks from alignments
- Handles gaps in alignments properly
- Computes position frequency matrices and consensus sequences
- Supports multiple weights and multiple peaks per weight
- Generates per-position nucleotide composition comparisons

**Usage:**
```bash
python extract_top_bottom_motifs.py \
    --variable BIO_11_6 \
    --phenotypes phenotypes.csv \
    --phenotype-id-column IID \
    --val-ids val_ids.txt \
    --weights weights_explanation.csv \
    --peaks gradient_peaks_summary.csv \
    --align-dir alignments/ \
    --output-prefix results/BIO_11_6 \
    --weight-mode abs \
    --peak-mode best_zscore \
    --window-half 5 \
    --top-n 10 \
    --bottom-n 10 \
    --num-weights 1
```

**Options:**
- `--weight-mode`: Selection criterion for influential weights
  - `abs` - Highest absolute weight magnitude (default)
  - `positive` - Most positive weights
  - `negative` - Most negative weights
- `--peak-mode`: Peak selection strategy
  - `best_zscore` - Peak with highest z-score (default)
  - `first` - First peak in file
  - `all` - Process all peaks for each weight
- `--window-half`: Half-window size around peak (total length = 2×half+1)
- `--num-weights`: Number of top weights to analyze

---

### Step 2: Mutation Embedding Generation

**Script:** `embed_sequences_mutations.py`  
**SLURM:** `run_embed_caduceus_mutations.sh`

Generates embeddings for mutated sequences at positions identified as important by gradient analysis. Supports single-position and multi-position combination mutations.

**Input:**
- Caduceus model path
- Selected weights CSV (from Step 1 or attribution analysis)
- Gradients directory (integrated gradients CSVs per SNP/dimension)
- Accessions list (one ID per line)
- Alignment directory (MSA FASTAs per SNP)
- Mutation parameters (number of positions, combinations, etc.)

**Output:**
- Per-weight subdirectories: `w{rank}_snp{snp}_dim{dim}/`
- Per-accession mutation files: `accession_{id}_mutations.npz`
  - `original` - Original embedding (1D array)
  - `mutations` - Single-position mutations (positions × alts × embed_dim)
  - `positions` - Ungapped position indices
  - `alignment_positions` - Alignment column indices
  - `bases` - Reference bases at each position
  - `alts` - Alternative bases tested
  - `combo_mutations` - Multi-position mutation embeddings (optional)
  - `combo_variant_starts` - Start indices in flattened combo arrays
  - `combo_variant_lengths` - Number of positions per combo
  - `combo_positions_flat` - Flattened position lists
  - `combo_ref_bases_flat` - Flattened reference bases
  - `combo_alt_bases_flat` - Flattened alternative bases
  - `metadata` - Mutation metadata (rank, SNP, dimension, etc.)
- `mutation_embeddings_manifest.csv` - Summary of all generated mutation files

**Key Features:**
- Z-score filtering (|z| > 1.96) to select significant gradient positions
- Bidirectional embedding (forward + reverse complement)
- Single-position mutations (all ACGT alternatives)
- Multi-position combination mutations (optional)
- Configurable combination size limits
- Safety caps on variant counts to prevent memory issues
- Heartbeat logging for long-running jobs
- Handles alignment gaps properly (skips gaps per accession)

**Usage:**
```bash
python embed_sequences_mutations.py \
    --model_name_or_path "kuleshov-group/caduceus-ph_seqlen-131k_d_model-256_n_layer-16" \
    --selected_weights selected_weights.csv \
    --gradients_dir gradients/ \
    --accessions_list accessions.txt \
    --alignment_dir alignments/ \
    --num-weights 50 \
    --top-grad-positions 8 \
    --mutation-output-dir mutations/ \
    --pool_method max \
    --mutate-combinations \
    --max-combination-size 3 \
    --combination-variant-limit 100000 \
    --no-ddp
```

**Mutation Strategies:**

**Single-position mutations:**
- For each selected position, mutate to all alternative bases (A→C,G,T)
- Each position processed independently
- Generates 3 variants per position (original base excluded)

**Multi-position combinations:**
- When `--mutate-combinations` enabled
- Generates all subset combinations from size 2 up to `--max-combination-size`
- For each subset, generates all Cartesian products of alternative bases
- Example: 2 positions with 3 alts each → 9 combination variants
- Safety limit: `--combination-variant-limit` prevents excessive memory use

**Options:**
- `--num-weights`: Number of top-ranked weights to process
- `--top-grad-positions`: Number of highest-gradient positions per weight
- `--mutation-window-context`: Flanking bases around mutation (0=full sequence)
- `--mutate-combinations`: Enable multi-position mutations
- `--max-combination-size`: Maximum positions in combination (≥2)
- `--combination-variant-limit`: Hard cap on total combination variants
- `--no-ddp`: Disable distributed training (use for single GPU)
- `--heartbeat-seconds`: Logging interval for long jobs (default: 300)

---

### Step 3: Mutation Effect Analysis

**Script:** `analyze_mutation_effects.py`  
**SLURM:** `run_analyze_mutation_effects.sh`

Analyzes how mutations affect model predictions by applying mutated embeddings to trained models and computing prediction changes.

**Input:**
- Original embeddings pickle (baseline predictions)
- Mutations root directory (from Step 2)
- Trained model pickle (with scaler and models)
- Target variable name
- Optional: accessions file to filter analysis

**Output Structure:**

**Main outputs (per direction):**
- `best_single_mutation_per_accession.csv` - Best single mutation per accession
- `best_combo_mutation_per_accession.csv` - Best combination mutation per accession
- `best_single_mutation_per_accession_non_{direction}.csv` - Opposite direction singles
- `best_combo_mutation_per_accession_non_{direction}.csv` - Opposite direction combos
- `variant_overlap_summary.csv` - Shared variants across accessions
- `per_snp_overlap_summary.csv` - Summary statistics per SNP

**Cumulative analysis (per top-N scenario):**
- Subdirectories: `1/`, `2/`, `3/`, ..., `10/`, `all/`
- `per_accession_cumulative_variants.csv` - Step-by-step variant application
- `per_accession_cumulative_deltas.csv` - Cumulative prediction changes
- `per_accession_cumulative_combo_variants.csv` - Combined singles+combos steps
- `per_accession_cumulative_combo_deltas.csv` - Combined cumulative changes

**Intermediate NPZ files (per accession):**
- `intermediates/{accession}.npz` - Detailed per-accession results
  - `baseline_pred` - Original prediction
  - `positive_single_variants` - Filtered single mutations (structured array)
  - `positive_combo_variants` - Filtered combo mutations (structured array)
  - `single_cumulative_steps` - Single-only cumulative application
  - `combined_cumulative_steps` - Combined cumulative application
  - `best_single_delta` - Best single mutation effect
  - `best_combo_delta` - Best combo mutation effect
  - `final_single_only_pred` - Final prediction (singles only)
  - `final_combined_pred` - Final prediction (singles + combos)
- `intermediates/intermediates_index.csv` - Index of all NPZ files

**Key Features:**
- Supports positive or negative effect analysis
- Best mutation per accession (highest/lowest delta)
- Variant overlap analysis (shared mutations across accessions)
- Cumulative multi-SNP simulation (applies mutations sequentially)
- Multiple top-N scenarios (1, 2, 3, ..., 10, all)
- Separate single-only and combined (singles+combos) analysis
- Per-accession intermediate files for detailed investigation
- Enforces one variant per SNP (no duplicate SNP mutations)
- Tracks step-by-step prediction changes

**Usage:**
```bash
python analyze_mutation_effects.py \
    --embeddings-pkl embeddings.pkl \
    --mutations-root mutations/ \
    --model-pkl model.pkl \
    --target-variable BIO_11_6 \
    --output-dir analysis/ \
    --top-n-snps 1 2 3 4 5 10 \
    --top-n-combined 1 2 3 4 5 10 \
    --effect-direction positive \
    --accessions-file val_ids.txt
```

**Effect Directions:**
- `positive` - Mutations that increase the target variable (default)
- `negative` - Mutations that decrease the target variable

**Cumulative Analysis Logic:**

1. **Filter mutations** by effect direction (positive/negative delta)
2. **Sort variants** by delta magnitude (best improvements/decreases first)
3. **Apply sequentially** to embedding matrix
4. **Track changes:**
   - Original single delta (isolated effect)
   - Step delta (marginal effect when added to previous mutations)
   - Cumulative delta (total effect from baseline)
   - Prediction after step
5. **Enforce constraints:**
   - One variant per SNP (first variant for each SNP is kept)
   - Stop at top-N unique SNPs

**Options:**
- `--top-n-snps`: List of top-N values for single-only analysis
- `--top-n-combined`: List of top-N values for combined analysis
- `--effect-direction`: `positive` or `negative` effects
- `--limit-snps`: Process only first N SNPs (for testing)
- `--accessions-file`: Restrict to specific accessions
- `--no-save-intermediates`: Disable per-accession NPZ files
- `--intermediates-dir`: Custom directory for intermediates

---

## Analysis Notebook

**Notebook:** `mutation_analysis.ipynb`

Jupyter notebook for interactive analysis and visualization of mutation effects. Includes:
- Loading and exploring mutation results
- Plotting cumulative effect curves
- Comparing single vs. combo mutations
- Analyzing variant overlap patterns
- Visualizing per-SNP effects
- Statistical summaries and distributions

---

## File Naming Conventions

### Python Scripts
- `extract_top_bottom_motifs.py` - Phenotype stratification and motif extraction
- `embed_sequences_mutations.py` - Generate mutated embeddings
- `analyze_mutation_effects.py` - Quantify mutation effects on predictions

### SLURM Scripts
- `run_extract_top_bottom_motifs.sh` - Submit motif comparison job
- `run_embed_caduceus_mutations.sh` - Submit mutation embedding job
- `run_analyze_mutation_effects.sh` - Submit mutation analysis job

### Output Files

**Motif comparison:**
- `*_motifs.fasta` - Sequence windows
- `*_pfm.tsv` - Position frequency matrices
- `*_consensus.txt` - Consensus sequences
- `motif_comparison.tsv` - Per-position deltas

**Mutation embeddings:**
- `accession_*_mutations.npz` - Per-accession mutation embeddings
- `mutation_embeddings_manifest.csv` - File inventory

**Effect analysis:**
- `best_*_mutation_per_accession.csv` - Best mutations per accession
- `variant_overlap_summary.csv` - Shared variants
- `per_snp_overlap_summary.csv` - SNP-level statistics
- `per_accession_cumulative_*.csv` - Cumulative analysis results
- `intermediates/*.npz` - Detailed per-accession data

---

## Configuration

### Common Parameters

**Models & Data:**
- `--model_name_or_path` - Caduceus model HuggingFace ID or path
- `--embeddings-pkl` - Original embeddings pickle file
- `--model-pkl` - Trained prediction model
- `--target-variable` - Target trait/variable name
- `--accessions_list` / `--accessions-file` - Accessions to process

**Mutation Selection:**
- `--num-weights` / `--top-n-snps` - Number of weights/SNPs to analyze
- `--top-grad-positions` - Positions per weight to mutate
- `--weight-mode` - abs/positive/negative (motif comparison)
- `--peak-mode` - best_zscore/first/all (motif comparison)

**Mutation Generation:**
- `--mutate-combinations` - Enable multi-position mutations
- `--max-combination-size` - Maximum positions per combination
- `--combination-variant-limit` - Safety cap on variants
- `--mutation-window-context` - Flanking bases around mutation

**Analysis:**
- `--effect-direction` - positive/negative effect analysis
- `--top-n-snps` - List of cumulative scenario sizes (singles)
- `--top-n-combined` - List of cumulative scenario sizes (combined)

### SLURM Resource Requirements

| Script | Partition | GPU | Memory | Time | Notes |
|--------|-----------|-----|--------|------|-------|
| extract_top_bottom_motifs | cpu | 0 | 16G | 2h | Fast, CPU-only |
| embed_sequences_mutations | gpu | 1 | 64G | 92h | GPU-intensive, long runtime |
| analyze_mutation_effects | cpu | 0 | 32G | 24h | CPU-only, memory for large datasets |

---

## Dependencies

### Python Packages
- `numpy` - Numerical operations
- `pandas` - Data manipulation
- `torch` - PyTorch for Caduceus model
- `transformers` - HuggingFace model loading
- `biopython` - FASTA parsing (motif comparison)
- `tqdm` - Progress bars
- `pickle` - Serialization
- `json` - Metadata storage

### External Tools
- **Caduceus model** - DNA foundation model for embeddings
- **CUDA** - GPU computation (for embedding generation)

### Singularity Containers
- `caduceus_align.sif` - Caduceus with alignment tools (motif comparison)
- `caduceus.sif` - Caduceus for mutation embedding (GPU)

---

## Output Results Structure

The `Results/` directory contains pre-computed analysis results organized by:

```
Results/
  AnalysisAll/
    Positive/              # Mutations that increase target variable
      {VARIABLE}/          # Per-variable results (e.g., BIO_11_6)
        best_single_mutation_per_accession.csv
        best_combo_mutation_per_accession.csv
        variant_overlap_summary.csv
        per_snp_overlap_summary.csv
        1/, 2/, ..., 10/   # Top-N cumulative scenarios
        intermediates/     # Per-accession NPZ files
    Negative/              # Mutations that decrease target variable
      {VARIABLE}/
        (same structure as Positive)
```

**Key result files:**

1. **Best mutations** - Single best variant per accession
2. **Overlap summaries** - Variants shared across multiple accessions
3. **Cumulative results** - Sequential application of multiple mutations
4. **Intermediates** - Detailed per-accession data for further analysis

---

## Workflow Integration

### Prerequisites from Attribution Analysis

Before running mutation analysis, complete these attribution pipeline steps:

1. **Model weight explanation** (`explain_model_weights.py`)
   - Generates `weights_explanation.csv` with important SNP/dimension pairs

2. **Saliency map generation** (`generate_saliency_maps.py`)
   - Generates `integrated_gradients_averaged_SNP*_dim*.csv` files
   - Creates alignment directory with MSA FASTAs

3. **Gradient peak extraction** (`extract_gradient_peaks.py`)
   - Generates `gradient_peaks_summary.csv` with peak positions

### Full Pipeline Example

```bash
# Step 0: Prerequisites (from attribution pipeline)
cd ../attribution
sbatch run_explain_model_weights.sh
sbatch run_saliency_maps.sh
sbatch run_extract_gradient_peaks.sh

# Step 1: Compare motifs between phenotype extremes
cd ../mutations
sbatch run_extract_top_bottom_motifs.sh

# Step 2: Generate mutation embeddings
sbatch run_embed_caduceus_mutations.sh

# Step 3: Analyze mutation effects
sbatch run_analyze_mutation_effects.sh

# Step 4: Explore results interactively
jupyter notebook mutation_analysis.ipynb
```

---

## Troubleshooting

### Common Issues

**1. "Alignment file not found"**
- Ensure saliency map generation completed with alignments
- Check alignment directory path matches expected structure
- Verify SNP indices in weights match alignment file names

**2. "Accession not in alignment"**
- Check accession ID format (string vs numeric)
- Verify accession IDs match between phenotype file and alignments
- Script attempts both string and numeric matching automatically

**3. "No positions pass z-score filter"**
- Gradient signals may be too weak for this weight
- Consider lowering z-score threshold (currently 1.96 for 95% CI)
- Check if gradients directory contains valid data

**4. "Combination variant limit exceeded"**
- Normal for large combination sizes with many alternatives
- Combinations are skipped when limit reached (single mutations still processed)
- Reduce `--max-combination-size` or increase `--combination-variant-limit`

**5. Memory issues in mutation embedding**
- Reduce `--num-weights` to process fewer weights
- Reduce `--top-grad-positions` for fewer positions per weight
- Lower `--max-combination-size` to reduce combination count
- Increase SLURM memory allocation

**6. "Gap in ungapped sequence"**
- Some accessions have gaps at selected positions
- These positions are automatically skipped for that accession
- This is expected behavior when using MSA alignments

### Debug Mode

Run scripts interactively for debugging:

```bash
# Interactive GPU node for mutation embedding
srun --partition=gpu --gres=gpu:1 --mem=64G --time=2:00:00 --pty bash
singularity shell --nv containers/caduceus.sif
python embed_sequences_mutations.py [args] --no-ddp

# Interactive CPU node for analysis
srun --mem=32G --time=2:00:00 --pty bash
singularity shell containers/caduceus_align.sif
python analyze_mutation_effects.py [args] --verbose
```

### Verification Steps

**Check mutation embeddings:**
```python
import numpy as np
npz = np.load('accession_1001_mutations.npz', allow_pickle=True)
print(f"Original shape: {npz['original'].shape}")
print(f"Mutations shape: {npz['mutations'].shape}")
print(f"Positions: {npz['positions']}")
print(f"Bases: {npz['bases']}")
if 'combo_mutations' in npz:
    print(f"Combo mutations: {npz['combo_mutations'].shape}")
```

**Check analysis results:**
```python
import pandas as pd
singles = pd.read_csv('best_single_mutation_per_accession.csv')
print(f"Accessions: {singles['accession'].nunique()}")
print(f"Mean delta: {singles['delta_pred'].mean():.4f}")
print(f"SNPs involved: {singles['snp_index'].nunique()}")
```

---

## Advanced Usage

### Processing Specific Accessions

Use accessions file to restrict analysis to specific individuals:

```bash
# Create accession list
echo "1001" > selected_accessions.txt
echo "1002" >> selected_accessions.txt
echo "1003" >> selected_accessions.txt

# Run with filter
python analyze_mutation_effects.py \
    --embeddings-pkl embeddings.pkl \
    --mutations-root mutations/ \
    --model-pkl model.pkl \
    --target-variable BIO_11_6 \
    --output-dir analysis/ \
    --accessions-file selected_accessions.txt
```

### Multiple Top-N Scenarios

Analyze multiple cumulative scenarios in one run:

```bash
python analyze_mutation_effects.py \
    --embeddings-pkl embeddings.pkl \
    --mutations-root mutations/ \
    --model-pkl model.pkl \
    --target-variable BIO_11_6 \
    --output-dir analysis/ \
    --top-n-snps 1 2 3 5 10 15 20 \
    --top-n-combined 1 2 3 5 10 15 20
```

Creates subdirectories: `1/`, `2/`, `3/`, `5/`, `10/`, `15/`, `20/` with cumulative results.

### Analyzing Negative Effects

To find mutations that decrease trait values:

```bash
python analyze_mutation_effects.py \
    --embeddings-pkl embeddings.pkl \
    --mutations-root mutations/ \
    --model-pkl model.pkl \
    --target-variable BIO_11_6 \
    --output-dir analysis_negative/ \
    --effect-direction negative \
    --top-n-snps 1 2 3 4 5
```

### Processing All Peaks

To analyze all peaks for each weight (not just best):

```bash
python extract_top_bottom_motifs.py \
    --variable BIO_11_6 \
    --phenotypes phenotypes.csv \
    --phenotype-id-column IID \
    --val-ids val_ids.txt \
    --weights weights_explanation.csv \
    --peaks gradient_peaks_summary.csv \
    --align-dir alignments/ \
    --output-prefix results/ \
    --peak-mode all \
    --num-weights 10
```

Creates per-peak outputs: `peak1_*`, `peak2_*`, `peak3_*`, etc.

---
