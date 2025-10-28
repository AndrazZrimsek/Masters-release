#!/bin/bash
#SBATCH --job-name=elasticnet_all_512
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output=logs/regression/linear_pipeline-%j.log
#SBATCH --error=logs/regression/linear_pipeline-%j.err

echo "Starting ML Pipeline for Environmental Variables Prediction..."
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Date: $(date)"

# Change to the project directory
cd /d/hpc/projects/arabidopsis_fri/Masters

# Run the ML pipeline script
# You can either run this directly with python if you have the required packages,
# or use a container that has scikit-learn, pandas, numpy installed

# Option 1: Direct python execution (if packages are available)
# python src/train_predict_models.py \
#     --embeddings_dir Embeddings \
#     --csv_files GWAS/coords_with_bioclim_30s_fixed.csv GWAS/coords_with_soil.csv \
#     --train_ids GWAS/train_ids.txt \
#     --val_ids GWAS/val_ids.txt \
#     --output_dir Results


        # --embeddings_dir /d/hpc/projects/arabidopsis_fri/Masters/Results/DeepSNP/filtered_embeddings_59234861 \

# Option 2: Using a container (uncomment if needed)
srun singularity exec --bind /d/hpc/projects/arabidopsis_fri:/d/hpc/projects/arabidopsis_fri \
    containers/container-pt.sif \
    python /d/hpc/projects/arabidopsis_fri/Masters/src/train_predict_models.py \
        --embeddings_dir /d/hpc/projects/arabidopsis_fri/Masters/Embeddings/deepsnp \
        --csv_files /d/hpc/projects/arabidopsis_fri/Masters/Data/combined_variables_dataset_normalized.csv \
        --train_ids /d/hpc/projects/arabidopsis_fri/Masters/GWAS/train_ids.txt \
        --val_ids /d/hpc/projects/arabidopsis_fri/Masters/GWAS/val_ids.txt \
        --output_dir /d/hpc/projects/arabidopsis_fri/Masters/Results/Report/DeepSNP/ \
        --model elasticnet \
        --job_id "$SLURM_JOB_ID" \
        --jobname "$SLURM_JOB_NAME" \
        # --average-embeddings \
        # --svd-dim 10

echo "ML Pipeline completed!"
echo "Check Results/ directory for output files"
echo "Date: $(date)"
