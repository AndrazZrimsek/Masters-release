#!/bin/bash
#SBATCH --job-name=compare_attr
#SBATCH --output=logs/compare_attributions_%j.out
#SBATCH --error=logs/compare_attributions_%j.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G

# Set up paths
MASTERS_DIR="/d/hpc/projects/arabidopsis_fri/Masters"
SCRIPT="${MASTERS_DIR}/src/compare_attributions.py"
ATTR_FILE="${MASTERS_DIR}/Results/Explainability/512_11_6/BIO_11_6/attributions_BIO_11_6_SNP75_dim16.csv"
IG_FILE="${MASTERS_DIR}/Results/Gradients/512_15_aligned_raw/BIO_11_6/integrated_gradients_averaged_SNP75_dim16.csv"
OUTPUT_DIR="${MASTERS_DIR}/Results/Report/figures/attribution_comparison"
OUTPUT_FILE="${OUTPUT_DIR}/attribution_comparison_SNP75_dim16_BIO_11_6_zscore"

# Create output directory
mkdir -p "${OUTPUT_DIR}"
mkdir -p "${MASTERS_DIR}/logs"

# Change to working directory
cd "${MASTERS_DIR}"

# Run the comparison script using singularity container
srun singularity exec containers/caduceus_align.sif python3 "${SCRIPT}" \
    --attr-file "${ATTR_FILE}" \
    --ig-file "${IG_FILE}" \
    --output "${OUTPUT_FILE}" \
    --norm-method zscore

echo "Attribution comparison completed. Output saved to: ${OUTPUT_FILE}"
