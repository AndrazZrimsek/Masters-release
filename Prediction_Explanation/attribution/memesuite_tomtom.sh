#!/bin/bash
#SBATCH --job-name=tomtom
#SBATCH --output=logs/enrichment/tomtom_%j.out
#SBATCH --error=logs/enrichment/tomtom_%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1

# Set paths (edit as needed)
SIF="/d/hpc/projects/arabidopsis_fri/Masters/containers/memesuite_latest.sif"
# FASTA="/d/hpc/projects/arabidopsis_fri/Masters/Data/Pseudogenomes/SparSNP_Joint_512/all_sequences.fasta"
MOTIFS="/d/hpc/projects/arabidopsis_fri/Masters/Results/Report/Explainability/SizeTest_500_512/GradientPeaks/SEA_Consensus/consensus_enriched5.meme"
# MOTIFS="/d/hpc/projects/arabidopsis_fri/Masters/Results/Report/Explainability/SizeTest_500_512/VariableAnalysis/Enriched/soil.meme"
DATABASE="/d/hpc/projects/arabidopsis_fri/Masters/Data/MotifDB/JASPAR2024_CORE_plants_non-redundant_v2.meme"
OUTDIR="/d/hpc/projects/arabidopsis_fri/Masters/Results/Report/Explainability/SizeTest_500_512/GradientPeaks/TomTom_Consensus5"

mkdir -p "$OUTDIR"
mkdir -p logs/enrichment

database_file=$(basename "$DATABASE")
echo "Running TomTom with motifs $MOTIFS and database $database_file"
echo "Output directory: $OUTDIR"

# Run TomTom using Singularity
srun singularity exec "$SIF" tomtom --oc "$OUTDIR" "$MOTIFS" "$DATABASE" 

echo "TomTom finished at $(date)"