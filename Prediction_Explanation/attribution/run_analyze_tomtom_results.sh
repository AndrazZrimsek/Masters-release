#!/bin/bash
#SBATCH --job-name=analyze_tomtom_results
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=0:30:00
#SBATCH --output=logs/tomtom/analyze_tomtom_results_%j.out
#SBATCH --error=logs/tomtom/analyze_tomtom_results_%j.err

set -e

echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "Working Directory: $(pwd)"

# Set paths
SCRIPT_DIR="/d/hpc/projects/arabidopsis_fri/Masters"
MOTIF_FILE="/d/hpc/projects/arabidopsis_fri/Masters/Results/Report/Explainability/SizeTest_500_512/GradientPeaks/SEA_Consensus/consensus_enriched.meme"
TOMTOM_TSV="/d/hpc/projects/arabidopsis_fri/Masters/Results/Report/Explainability/SizeTest_500_512/GradientPeaks/TomTom_Consensus5/tomtom.tsv"
OUTPUT_DIR="/d/hpc/projects/arabidopsis_fri/Masters/Results/Report/Explainability/SizeTest_500_512/GradientPeaks/TomTom_Consensus5/Analysis01"

mkdir -p "$OUTPUT_DIR"
mkdir -p logs/tomtom

cd "$SCRIPT_DIR"

echo "Running Tomtom analysis..."
srun singularity exec containers/memelite.sif python src/analyze_tomtom_results.py \
    --meme_file "$MOTIF_FILE" \
    --tomtom_tsv "$TOMTOM_TSV" \
    --output_dir "$OUTPUT_DIR" \
    --qval_thresh 0.1

if [ $? -eq 0 ]; then
    echo "Tomtom analysis completed successfully!"
    if [ -f "$OUTPUT_DIR/tomtom_analysis_report.txt" ]; then
        echo "Analysis report generated: $OUTPUT_DIR/tomtom_analysis_report.txt"
    else
        echo "Warning: Analysis report not found."
    fi
else
    echo "Error: Tomtom analysis failed!"
    exit 1
fi

echo "End Time: $(date)"
echo "Job completed."
