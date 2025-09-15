#!/bin/bash
#SBATCH -p hbm-long-96core
#SBATCH -t 48:00:00
#SBATCH --output=res.txt

python train_rf.py \
    --model_name per_antibiotic_models_v3_both_dev \
    --grouping per_species \
    --train_on dev \
    --feature_type both \
    --dataset_dir /gpfs/scratch/jvaska/CAMDA_AMR/AMR_v2/dnabert/inference/outputs/rf_datasets/per_antibiotic_models_v3/per_antibiotic/train \
    --eval True
