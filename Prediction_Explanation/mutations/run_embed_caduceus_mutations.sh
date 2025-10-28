#!/bin/bash
#SBATCH --job-name=embed_caduceus_mut
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --exclude=gwn07
#SBATCH --mem=64G
#SBATCH --time=92:00:00
#SBATCH --output=logs/embed/cadu/mutations/embed_caduceus_mut-%j.log
#SBATCH --error=logs/embed/cadu/mutations/embed_caduceus_mut-%j.err

export RANK=0
export WORLD_SIZE=1
export MASTER_ADDR=localhost
export MASTER_PORT=29321

        # --seq_len 10000 \

#############################################
# User-configurable inputs (edit these paths)
#############################################

# Model + pooling
MODEL_NAME="kuleshov-group/caduceus-ph_seqlen-131k_d_model-256_n_layer-16"
POOL_METHOD="max"   # or "avg"
TARGET_VAR="BIO1"

# Core mutation inputs
SELECTED_WEIGHTS_CSV="/d/hpc/projects/arabidopsis_fri/Masters/Results/Report/Explainability/SizeTest_500_512/MotifCompare/${TARGET_VAR}_50/selected_weights.csv"   # ranked (rank,snp_index,embedding_dim,...)
GRADIENTS_DIR="/d/hpc/projects/arabidopsis_fri/Masters/Results/Report/Explainability/SizeTest_500_512/Gradients/${TARGET_VAR}"                              # per-SNP/embedding_dim gradient CSVs
ACCESSIONS_LIST="/d/hpc/projects/arabidopsis_fri/Masters/Results/Report/Explainability/SizeTest_500_512/MotifCompare/all_accessions.txt"                     # one accession ID per line
ALIGNMENT_DIR="/d/hpc/projects/arabidopsis_fri/Masters/Results/Report/Explainability/SizeTest_500_512/Gradients/alignments"                                # alignment FASTA files per SNP
MUTATION_OUTPUT_DIR="/d/hpc/projects/arabidopsis_fri/Masters/Results/Report/Mutations/Combined/All/${TARGET_VAR}"                     # output NPZ + manifest

# Selection / enumeration controls
NUM_WEIGHTS=50                 # how many top weights (rows) to process
TOP_GRAD_POS=8                  # number of top gradient positions per (weight,SNP,embedding_dim)
WINDOW_CONTEXT=0               # flank bp radius around each mutated position (set 0 for full seq if supported)

# Combination mutation controls
MUTATE_COMBINATIONS=1           # 1 to enable multi-position combinations, 0 to disable
MAX_COMBINATION_SIZE=3          # largest subset size (>=2) when MUTATE_COMBINATIONS=1
COMBINATION_VARIANT_LIMIT=100000 # hard cap on total combination variants (skip extras)

# Misc runtime
BP_PER_TOKEN=1                  # keep consistent with model training
EXTRA_ARGS=""                  # append any experimental flags here

echo "Preparing directories..."
mkdir -p logs/embed/cadu/mutations "$MUTATION_OUTPUT_DIR"

echo "Running Caduceus mutation embedding script..."

# Build combination flag string
COMBO_FLAGS=""
if [[ "$MUTATE_COMBINATIONS" == "1" ]]; then
        COMBO_FLAGS="--mutate-combinations --max-combination-size $MAX_COMBINATION_SIZE --combination-variant-limit $COMBINATION_VARIANT_LIMIT"
fi

srun singularity exec --nv containers/caduceus.sif bash -c "
                export RANK=0
                export WORLD_SIZE=1
                export MASTER_ADDR=localhost
                export MASTER_PORT=29321
                python /d/hpc/projects/arabidopsis_fri/Masters/caduceus/embed_sequences_mutations.py \
                                --model_name_or_path $MODEL_NAME \
                                --pool_method $POOL_METHOD \
                                --bp_per_token $BP_PER_TOKEN \
                                --selected_weights $SELECTED_WEIGHTS_CSV \
                                --gradients_dir $GRADIENTS_DIR \
                                --accessions_list $ACCESSIONS_LIST \
                                --alignment_dir $ALIGNMENT_DIR \
                                --num-weights $NUM_WEIGHTS \
                                --top-grad-positions $TOP_GRAD_POS \
                                --mutation-output-dir $MUTATION_OUTPUT_DIR \
                                --mutation-window-context $WINDOW_CONTEXT \
                                $COMBO_FLAGS \
                                $EXTRA_ARGS \
                                --no-ddp 
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