#!/usr/bin/env bash
#SBATCH --job-name=mut_effects
#SBATCH --output=logs/mut_effects/mut_effects-%j.log
#SBATCH --error=logs/mut_effects/mut_effects-%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

set -euo pipefail

mkdir -p logs/mut_effects

# Ensure unbuffered python output
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=$OMP_NUM_THREADS
export OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS
export NUMEXPR_NUM_THREADS=$OMP_NUM_THREADS

# Adjust path to singularity image if needed
SIF=/d/hpc/projects/arabidopsis_fri/Masters/containers/caduceus_align.sif

if [ ! -f "$SIF" ]; then
  echo "Singularity image not found: $SIF" >&2
  exit 1
fi

### Static configuration (edit these variables instead of passing args) ###
# TARGET_VAR="BIO1"
# TARGET_VAR="BIO_10_5"
TARGET_VAR="BIO_11_6"
EMBED_PKL="/d/hpc/projects/arabidopsis_fri/Masters/Embeddings/size_test/512/Caduceus_FixedLen_500_512.pkl"
MUT_ROOT="/d/hpc/projects/arabidopsis_fri/Masters/Results/Report/Mutations/Combined/All/${TARGET_VAR}"
MODEL_PKL="/d/hpc/projects/arabidopsis_fri/Masters/Results/Report/SizeTest/size_test_512_500_60048928/elasticnet_model_Caduceus_FixedLen_500_512.pkl"
# OUT_DIR="/d/hpc/projects/arabidopsis_fri/Masters/Results/Report/Mutations/Combined/Analysis/TopPos/${TARGET_VAR}"
OUT_DIR="/d/hpc/projects/arabidopsis_fri/Masters/Results/Report/Mutations/Combined/All/Analysis/Negative/${TARGET_VAR}"
# OUT_DIR="/d/hpc/projects/arabidopsis_fri/Masters/Results/Report/Mutations/Combined/Analysis/AveragePos/${TARGET_VAR}"
# OUT_DIR="/d/hpc/projects/arabidopsis_fri/Masters/Results/Report/Mutations/Combined/Analysis/AverageNeg/${TARGET_VAR}"
# ACCESSIONS_FILE="/d/hpc/projects/arabidopsis_fri/Masters/Results/Report/Explainability/SizeTest_500_512/MotifCompare/top_accessions.txt"  # Optional: set path to file with accessions (one per line)
ACCESSIONS_FILE="/d/hpc/projects/arabidopsis_fri/Masters/Results/Report/Explainability/SizeTest_500_512/MotifCompare/all_accessions.txt"
# ACCESSIONS_FILE="/d/hpc/projects/arabidopsis_fri/Masters/Results/Report/Explainability/SizeTest_500_512/MotifCompare/average_accessions.txt"

echo "Configuration:";
echo "  EMBED_PKL=${EMBED_PKL}";
echo "  MUT_ROOT=${MUT_ROOT}";
echo "  MODEL_PKL=${MODEL_PKL}";
echo "  TARGET_VAR=${TARGET_VAR}";
echo "  OUT_DIR=${OUT_DIR}";
# echo "  TOP_N=${TOP_N}";
if [ -n "$ACCESSIONS_FILE" ]; then echo "  ACCESSIONS_FILE=${ACCESSIONS_FILE}"; fi

for req in EMBED_PKL MUT_ROOT MODEL_PKL TARGET_VAR OUT_DIR; do
  if [ -z "${!req}" ]; then
    echo "Missing required variable $req" >&2; exit 1; fi; done

mkdir -p "${OUT_DIR}" logs

srun singularity exec "$SIF" python src/analyze_mutation_effects.py \
  --embeddings-pkl "${EMBED_PKL}" \
  --mutations-root "${MUT_ROOT}" \
  --model-pkl "${MODEL_PKL}" \
  --target-variable "${TARGET_VAR}" \
  --output-dir "${OUT_DIR}" \
  --top-n-snps 1 2 3 4 5 6 7 8 9 10 \
  --top-n-combined 1 2 3 4 5 6 7 8 9 10 \
  --effect-direction "negative" \
  ${ACCESSIONS_FILE:+--accessions-file "$ACCESSIONS_FILE"}
