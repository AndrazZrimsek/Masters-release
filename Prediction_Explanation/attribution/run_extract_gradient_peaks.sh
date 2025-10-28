#!/bin/bash
#SBATCH --job-name=extract_gradient_peaks
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=2:00:00
#SBATCH --output=logs/gradient_peaks/gradient_peaks_%j.out
#SBATCH --error=logs/gradient_peaks/gradient_peaks_%j.err

# Exit on any error
set -e

# Print job information
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "Working Directory: $(pwd)"

# Set paths
GRADIENTS_DIR="/d/hpc/projects/arabidopsis_fri/Masters/Results/Report/Explainability/SizeTest_500_512/Gradients"
FASTA_DIR="/d/hpc/projects/arabidopsis_fri/Masters/Data/Pseudogenomes/SizeTest/SparSNP_FixedLen_500_512"
OUTPUT_DIR="/d/hpc/projects/arabidopsis_fri/Masters/Results/Report/Explainability/SizeTest_500_512/GradientPeaks"
SCRIPT_DIR="/d/hpc/projects/arabidopsis_fri/Masters"

# Create output directory and logs
mkdir -p "$OUTPUT_DIR"
mkdir -p logs/gradient_peaks

# Check required directories exist
if [ ! -d "$GRADIENTS_DIR" ]; then
    echo "Error: Gradients directory not found: $GRADIENTS_DIR"
    echo "Please run generate_saliency_maps.py first to generate gradient files"
    exit 1
fi

if [ ! -d "$FASTA_DIR" ]; then
    echo "Error: FASTA directory not found: $FASTA_DIR"
    exit 1
fi

# Check if gradient CSV files exist
CSV_COUNT=$(find "$GRADIENTS_DIR" -name "integrated_gradients_averaged_SNP*_dim*.csv" 2>/dev/null | wc -l)
if [ "$CSV_COUNT" -eq 0 ]; then
    echo "Error: No integrated gradients CSV files found in $GRADIENTS_DIR"
    echo "Please run generate_saliency_maps.py first to generate gradient files"
    exit 1
fi

# Print configuration
echo "Configuration:"
echo "  Gradients Directory: $GRADIENTS_DIR"
echo "  FASTA Directory: $FASTA_DIR"
echo "  Output Directory: $OUTPUT_DIR"
echo "  Found CSV files: $CSV_COUNT"
echo ""

# Change to script directory
cd "$SCRIPT_DIR"

# Run the extraction script using singularity
echo "Starting gradient peaks extraction..."
srun singularity exec containers/caduceus.sif python src/extract_gradient_peaks.py \
    --gradients_dir "$GRADIENTS_DIR" \
    --fasta_dir "$FASTA_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --window_size 11 

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo "Gradient peaks extraction completed successfully!"
    echo "Results saved to: $OUTPUT_DIR"
    
    # Count output files
    if [ -f "$OUTPUT_DIR/gradient_peaks_summary.csv" ]; then
        PEAK_COUNT=$(tail -n +2 "$OUTPUT_DIR/gradient_peaks_summary.csv" | wc -l)
        echo "Generated files:"
        echo "  gradient_peaks_summary.csv: $PEAK_COUNT gradient peaks"
        if [ -f "$OUTPUT_DIR/consensus_logos_at_peaks.png" ]; then
            echo "  consensus_logos_at_peaks.png: Consensus logo visualization"
        fi
    else
        echo "Warning: Expected output file not found"
    fi
    
else
    echo ""
    echo "Error: Gradient peaks extraction failed!"
    exit 1
fi

echo "End Time: $(date)"
echo "Job completed."
