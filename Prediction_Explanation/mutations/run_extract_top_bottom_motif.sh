#!/bin/bash
#SBATCH --job-name=motif_compare
#SBATCH --output=logs/motif_compare/motif_compare-%j.out
#SBATCH --error=logs/motif_compare/motif_compare-%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

set -euo pipefail

############################################
# User-configurable parameters (EDIT THESE)
############################################
VARIABLE="BIO_11_6"                                # Target variable name
PHENOTYPES_CSV="/d/hpc/projects/arabidopsis_fri/Masters/Data/combined_variables_dataset_normalized.csv"        # Phenotype table with accession IDs and variable column
PHENOTYPE_ID_COLUMN="IID"             # Column in phenotype table holding accession IDs
VALIDATION_IDS_FILE="/d/hpc/projects/arabidopsis_fri/Masters/GWAS/val_ids.txt"      # Validation accession IDs (newline / whitespace separated)
WEIGHTS_CSV="Results/Report/Explainability/SizeTest_500_512/weights_explanation.csv"
PEAKS_CSV="Results/Report/Explainability/SizeTest_500_512/GradientPeaks/gradient_peaks_summary.csv"
ALIGN_DIR="Results/Report/Explainability/SizeTest_500_512/Gradients/alignments"
OUTPUT_PREFIX="Results/Report/Explainability/SizeTest_500_512/MotifCompare/${VARIABLE}_50"
WEIGHT_MODE="abs"               # abs | positive | negative
PEAK_MODE="all"         # best_zscore | first | all
WINDOW_HALF=5                   # Half-window size (total length = 2*WINDOW_HALF+1)
TOP_N=10                         # Number of top accessions
BOTTOM_N=10                      # Number of bottom accessions

# Singularity container with Python + pandas + numpy
CONTAINER="containers/caduceus_align.sif"             # Adjust if different container needed
SCRIPT="src/compare_top_bottom_motif.py"       # Python script path
PYTHON_OPTS=""                                  # Extra python flags if desired

############################################
# Derived setup
############################################
mkdir -p logs/motif_compare
mkdir -p "$(dirname "$OUTPUT_PREFIX")"

echo "[$(date)] Motif comparison job starting"
echo "Variable: $VARIABLE"
echo "Weights file: $WEIGHTS_CSV"
echo "Peaks file: $PEAKS_CSV"
echo "Alignment dir: $ALIGN_DIR"
echo "Output prefix: $OUTPUT_PREFIX"

CMD=(
  python "$SCRIPT"
    --variable "$VARIABLE"
    --phenotypes "$PHENOTYPES_CSV"
    --phenotype-id-column "$PHENOTYPE_ID_COLUMN"
    --val-ids "$VALIDATION_IDS_FILE"
    --weights "$WEIGHTS_CSV"
    --peaks "$PEAKS_CSV"
    --align-dir "$ALIGN_DIR"
    --weight-mode "$WEIGHT_MODE"
    --peak-mode "$PEAK_MODE"
    --window-half "$WINDOW_HALF"
    --top-n "$TOP_N"
    --bottom-n "$BOTTOM_N"
    --output-prefix "$OUTPUT_PREFIX"
    --num-weights 50
)

echo "Executing command:"
printf ' %q' "${CMD[@]}"; echo

if [[ -f "$CONTAINER" ]]; then
  echo "Running inside container: $CONTAINER"
  singularity exec "$CONTAINER" "${CMD[@]}"
else
  echo "WARNING: Container $CONTAINER not found; running on host environment."
  "${CMD[@]}"
fi

echo "[$(date)] Motif comparison job finished"
