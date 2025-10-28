#!/bin/bash
#SBATCH --job-name=sea_consensus
#SBATCH --output=logs/enrichment/sea_consensus_%j.out
#SBATCH --error=logs/enrichment/sea_consensus_%j.err
#SBATCH --time=00:20:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1

set -euo pipefail

# Defaults (edit as needed)
SCRIPT_DIR="/d/hpc/projects/arabidopsis_fri/Masters"
SEA_BASE_DIR="/d/hpc/projects/arabidopsis_fri/Masters/Results/Report/Explainability/SizeTest_500_512/GradientPeaks/SEA_Consensus"
MOTIF_FILE="/d/hpc/projects/arabidopsis_fri/Masters/Results/Report/Explainability/SizeTest_500_512/GradientPeaks/gradient_peak_motifs_nonredundant.meme"
OUTPUT_FILE="/d/hpc/projects/arabidopsis_fri/Masters/Results/Report/Explainability/SizeTest_500_512/GradientPeaks/SEA_Consensus/consensus_enriched5.meme"
RATIO=5
SCORE_THR=10
AGGREGATE="majority"   # any | majority | all
MIN_SUPPORT=5           # for majority, require >=6/10 by default

usage() {
    echo "Usage: $0 [--sea-dir DIR] [--meme FILE] [--out FILE] [--ratio X] [--score X] [--aggregate any|majority|all] [--min-support N]" >&2
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --sea-dir) SEA_BASE_DIR="$2"; shift 2;;
        --meme) MOTIF_FILE="$2"; shift 2;;
        --out) OUTPUT_FILE="$2"; shift 2;;
        --ratio) RATIO="$2"; shift 2;;
        --score|--score-thresh|--score_thresh) SCORE_THR="$2"; shift 2;;
        --aggregate) AGGREGATE="$2"; shift 2;;
        --min-support|--min_support) MIN_SUPPORT="$2"; shift 2;;
        -h|--help) usage; exit 0;;
        *) echo "Unknown argument: $1" >&2; usage; exit 1;;
    esac
done

mkdir -p logs/enrichment
cd "$SCRIPT_DIR"

echo "Running consensus filtering over directory: $SEA_BASE_DIR"
srun singularity exec containers/memelite.sif python src/filter_significant_motifs.py \
    --meme_file "$MOTIF_FILE" \
    --sea_dir "$SEA_BASE_DIR" \
    --output_file "$OUTPUT_FILE" \
    --ratio "$RATIO" \
    --score_thresh "$SCORE_THR" \
    --aggregate "$AGGREGATE" \
    --min_support "$MIN_SUPPORT"

echo "Consensus filtering completed. Output: $OUTPUT_FILE"
echo "Job completed at $(date)"
