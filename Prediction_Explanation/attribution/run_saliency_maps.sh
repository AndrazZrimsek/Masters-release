#!/bin/bash
#SBATCH --job-name=integrated_gradients
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=96:00:00
#SBATCH --output=logs/gradients/gradients_%j.out
#SBATCH --error=logs/gradients/gradients_%j.err

# Exit on any error
set -e

# Print job information
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "Working Directory: $(pwd)"

# Load modules or activate environment
# Uncomment and modify as needed for your system
# module load cuda/11.8
# module load python/3.9

# Activate virtual environment if needed
# source /path/to/your/venv/bin/activate

# Set paths
CADUCEUS_MODEL_PATH="kuleshov-group/caduceus-ph_seqlen-131k_d_model-256_n_layer-16" 
FASTA_DIR="/d/hpc/projects/arabidopsis_fri/Masters/Data/Pseudogenomes/SizeTest/SparSNP_FixedLen_500_512"
VAL_IDS="/d/hpc/projects/arabidopsis_fri/Masters/GWAS/val_ids.txt"
WEIGHTS_FILE="/d/hpc/projects/arabidopsis_fri/Masters/Results/Report/Explainability/SizeTest_500_512/weights_explanation.csv"
OUTPUT_DIR="/d/hpc/projects/arabidopsis_fri/Masters/Results/Report/Explainability/SizeTest_500_512/Gradients"
SCRIPT_DIR="/d/hpc/projects/arabidopsis_fri/Masters"

# Create output directory
mkdir -p "$OUTPUT_DIR"
mkdir -p logs/gradients

# Check required files exist
if [ ! -f "$VAL_IDS" ]; then
    echo "Error: Validation IDs file not found: $VAL_IDS"
    exit 1
fi

if [ ! -f "$WEIGHTS_FILE" ]; then
    echo "Error: Weights file not found: $WEIGHTS_FILE"
    echo "Please run explain_model_weights.py first to generate the weights file"
    exit 1
fi

if [ ! -d "$FASTA_DIR" ]; then
    echo "Error: FASTA directory not found: $FASTA_DIR"
    exit 1
fi

# Print configuration
echo "Configuration:"
echo "  Caduceus Model: $CADUCEUS_MODEL_PATH"
echo "  FASTA Directory: $FASTA_DIR"
echo "  Validation IDs: $VAL_IDS"
echo "  Weights File: $WEIGHTS_FILE"
echo "  Output Directory: $OUTPUT_DIR"
echo ""

# Change to script directory
cd "$SCRIPT_DIR"

# Check GPU availability
echo "GPU Information:"
nvidia-smi
echo ""

singularity exec --nv containers/caduceus_align.sif python src/generate_saliency_maps.py \
    --caduceus_model_path "$CADUCEUS_MODEL_PATH" \
    --fasta_dir "$FASTA_DIR" \
    --val_ids "$VAL_IDS" \
    --weights_file "$WEIGHTS_FILE" \
    --output_dir "$OUTPUT_DIR" \
    --top_n_weights 50 \
    --generate_plots \
    --pooling max \
    --target_variable "BIO_11_6" \

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo "Saliency map generation completed successfully!"
    echo "Results saved to: $OUTPUT_DIR"
    
    # Count output files
    CSV_COUNT=$(find "$OUTPUT_DIR" -name "saliency_*.csv" | wc -l)
    PLOT_COUNT=$(find "$OUTPUT_DIR" -name "saliency_*_plot.png" | wc -l)
    
    echo "Generated files:"
    echo "  CSV files: $CSV_COUNT"
    echo "  Plot files: $PLOT_COUNT"
    
else
    echo ""
    echo "Error: Saliency map generation failed!"
    exit 1
fi

echo "End Time: $(date)"
echo "Job completed."
