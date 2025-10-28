#!/bin/bash
#SBATCH --job-name=embed_caduceus
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --exclude=gwn07
#SBATCH --mem=128G
#SBATCH --time=92:00:00
#SBATCH --output=logs/embed/cadu/embed_caduceus-%j.log
#SBATCH --error=logs/embed/cadu/embed_caduceus-%j.err

export RANK=0
export WORLD_SIZE=1
export MASTER_ADDR=localhost
export MASTER_PORT=29321

        # --seq_len 10000 \

echo "Running Caduceus embedding script..."

srun singularity exec --nv containers/caduceus.sif bash -c "
    export RANK=0
    export WORLD_SIZE=1
    export MASTER_ADDR=localhost
    export MASTER_PORT=29321
    python ./embed_sequences.py \
        --model_name_or_path kuleshov-group/caduceus-ph_seqlen-131k_d_model-256_n_layer-16 \
        --bp_per_token 1 \
        --downstream_save_dir ./outputs/downstream/test_embeddings \
        --name caduceus-ph_downstream-seqlen=131k \
        --accessions_dir /d/hpc/projects/arabidopsis_fri/Masters/Data/Pseudogenomes/DeepSNP/SparSNP_FixedLen_512 \
        --output_file /d/hpc/projects/arabidopsis_fri/Masters/Embeddings/DeepSNP/512_288/Caduceus_DeepSNP_288_512.pkl \
        --pool_method max
"

        # --join_regions \

# srun --gres=gpu:1 \
#      --export=ALL,NVIDIA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES,TMPDIR=/d/hpc/projects/arabidopsis_fri/tmp \
#      singularity exec \
#      --nv \
#      --containall \
#      --no-home \
#      --bind /d/hpc/projects/arabidopsis_fri \
#      containers/caduceus.sif \
#      bash -c "
#          export RANK=0
#          export WORLD_SIZE=1
#          export MASTER_ADDR=localhost
#          export MASTER_PORT=29321
#          python /d/hpc/projects/arabidopsis_fri/Masters/caduceus/embed_sequences.py \
#              --model_name_or_path kuleshov-group/caduceus-ph_seqlen-131k_d_model-256_n_layer-16 \
#              --seq_len 131072 \
#              --bp_per_token 1 \
#              --downstream_save_dir /d/hpc/projects/arabidopsis_fri/Masters/outputs/downstream/test_embeddings \
#              --name caduceus-ph_downstream-seqlen=131k \
#              --accessions_dir /d/hpc/projects/arabidopsis_fri/Masters/Data/Pseudogenomes/SparSNP_New_1000 \
#              --output_file /d/hpc/projects/arabidopsis_fri/Masters/Embeddings/SparSNP_1000.pkl
#      "