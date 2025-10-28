#!/bin/bash
#SBATCH --job-name=sea
#SBATCH --output=logs/enrichment/sea_%A_%a.out
#SBATCH --error=logs/enrichment/sea_%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --array=1-10

# Set paths (edit as needed)
SIF="/d/hpc/projects/arabidopsis_fri/Masters/containers/memesuite_latest.sif"
FASTA="/d/hpc/projects/arabidopsis_fri/Masters/Data/Pseudogenomes/SizeTest/SparSNP_FixedLen_500_512/all_sequences.fasta"
# FASTA="/d/hpc/projects/arabidopsis_fri/Masters/Results/Consensus/10000_consensus_sequences.fasta"
MOTIFS="/d/hpc/projects/arabidopsis_fri/Masters/Results/Report/Explainability/SizeTest_500_512/GradientPeaks/gradient_peak_motifs_nonredundant.meme"
# MOTIFS="/d/hpc/projects/arabidopsis_fri/Masters/Results/AttributionAnalysisLasso/weight_attribution/temp_positive_enriched/primary_motifs.meme"

# OUTDIR="/d/hpc/projects/arabidopsis_fri/Masters/Results/Enrichment/sea_512_lasso"
# PRIMARY="/d/hpc/projects/arabidopsis_fri/Masters/Results/Report/Pseudogenomes/BIO1_high.fasta"
# CONTROL="/d/hpc/projects/arabidopsis_fri/Masters/Results/Report/Pseudogenomes/BIO1_low.fasta"
BASE_OUTDIR="/d/hpc/projects/arabidopsis_fri/Masters/Results/Report/Explainability/SizeTest_500_512/GradientPeaks/SEA_Consensus"

SEED=${SLURM_ARRAY_TASK_ID:-1}
OUTDIR="$BASE_OUTDIR/seed_${SEED}"
mkdir -p "$OUTDIR"
mkdir -p logs/enrichment

echo "Running SEA (seed=$SEED) on $FASTA with motifs $MOTIFS"
echo "Output directory: $OUTDIR"

# Run SEA using Singularity
srun singularity exec "$SIF" sea \
	--oc "$OUTDIR" \
	--seed "$SEED" \
	--p "$FASTA" \
	--m "$MOTIFS" # --n "$PRIMARY"

echo "SEA finished at $(date)"