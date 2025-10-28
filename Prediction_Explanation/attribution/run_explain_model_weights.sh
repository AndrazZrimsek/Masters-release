#!/bin/bash
#SBATCH --job-name=explain_report_1000
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --exclude=gwn07
#SBATCH --mem=64G
#SBATCH --output=logs/explain/explain_weights_attr_%j.log
#SBATCH --error=logs/explain/explain_weights_attr_%j.err
#SBATCH --time=96:00:00

# MODEL_PKL="/d/hpc/projects/arabidopsis_fri/Masters/Results/Linear/SeqLen/linear_pipeline_fixed_100_58930672/elasticnet_model_Caduceus_FixedLen_1000_avg.pkl"
MODEL_PKL="/d/hpc/projects/arabidopsis_fri/Masters/Results/Report/SizeTest/size_test_512_500_60048928/elasticnet_model_Caduceus_FixedLen_500_512.pkl"
EMBEDDING_DIM=256
OUTPUT="/d/hpc/projects/arabidopsis_fri/Masters/Results/Report/Explainability/SizeTest_500_512/weights_explanation.csv"
CADUCEUS_MODEL_PATH="kuleshov-group/caduceus-ph_seqlen-131k_d_model-256_n_layer-16"
VAL_IDS="/d/hpc/projects/arabidopsis_fri/Masters/GWAS/val_ids.txt"
FASTA_DIR="/d/hpc/projects/arabidopsis_fri/Masters/Data/Pseudogenomes/SizeTest/SparSNP_FixedLen_500_512"
TOP_N=50
ATTRIBUTION_STEPS=100

# Create output directories if they don't exist
mkdir -p logs/explain
mkdir -p /d/hpc/projects/arabidopsis_fri/Masters/Results/Report/Explainability/SizeTest_500_512

# export PYTHONPATH=/d/hpc/projects/arabidopsis_fri:$PYTHONPATH

# Use srun and singularity container as in test_caduceus.sh
srun singularity exec --nv containers/caduceus.sif bash -c "\
    export PYTHONPATH=/d/hpc/projects/arabidopsis_fri:\$PYTHONPATH
    python /d/hpc/projects/arabidopsis_fri/Masters/src/explain_model_weights.py \
        --model '$MODEL_PKL' \
        --embedding_dim $EMBEDDING_DIM \
        --output '$OUTPUT' \
        --caduceus_model_path '$CADUCEUS_MODEL_PATH' \
        --val_ids '$VAL_IDS' \
        --top_n $TOP_N \
        --fasta_dir '$FASTA_DIR' \
        --compute_attributions \
        --attribution_steps $ATTRIBUTION_STEPS \
        --save_attributions \
        --force \
        --explain_only \
"

# To run WITHOUT attribution analysis, comment out the attribution flags:
# --compute_attributions \
# --attribution_steps $ATTRIBUTION_STEPS \
# --save_attributions \

# To force recomputation of existing files, add:
# --force \
