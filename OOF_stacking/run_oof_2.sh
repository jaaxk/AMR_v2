#!/bin/bash
#SBATCH -p a100-long
#SBATCH --output res.txt
#SBATCH --gres=gpu:1
#SBATCH --time 48:00:00

python -u oof_2.py \
    --data_base_dir /gpfs/scratch/jvaska/CAMDA_AMR/AMR_v2/data_pipeline/datasets/sequence_based/per_species/test/full_sequence_dataset.csv \
    --models_base_dir /gpfs/scratch/jvaska/CAMDA_AMR/AMR_v2/dnabert/finetune/finetuned_models/oof_run1 \
    --run_name run1_testset \
    --grouping full

