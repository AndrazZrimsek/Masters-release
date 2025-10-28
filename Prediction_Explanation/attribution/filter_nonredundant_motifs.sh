#!/bin/bash
#SBATCH --job-name=filter_motifs
#SBATCH --output=logs/enrichment/filter_motifs_%j.out
#SBATCH --error=logs/enrichment/filter_motifs_%j.err
#SBATCH --time=00:30:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1

# Set paths
SIF="/d/hpc/projects/arabidopsis_fri/Masters/containers/caduceus.sif"
SCRIPT="/d/hpc/projects/arabidopsis_fri/Masters/src/filter_nonredundant_motifs.py"
TOMTOM_TSV="/d/hpc/projects/arabidopsis_fri/Masters/Results/Report/Explainability/SizeTest_500_512_lasso/GradientPeaks/TomTomSelf/tomtom.tsv"
MEME_IN="/d/hpc/projects/arabidopsis_fri/Masters/Results/Report/Explainability/SizeTest_500_512_lasso/GradientPeaks/gradient_peak_motifs.meme"
MEME_OUT="/d/hpc/projects/arabidopsis_fri/Masters/Results/Report/Explainability/SizeTest_500_512_lasso/GradientPeaks/gradient_peak_motifs_nonredundant.meme"

mkdir -p logs/enrichment

echo "Running filter_nonredundant_motifs.py in Singularity container"
srun singularity exec "$SIF" python "$SCRIPT" \
    --tomtom_tsv "$TOMTOM_TSV" \
    --meme_in "$MEME_IN" \
    --meme_out "$MEME_OUT"

echo "Done at $(date)"
