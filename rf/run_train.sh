#!/bin/bash
#SBATCH -p hbm-long-96core
#SBATCH -t 48:00:00
#SBATCH --output=res.txt

python train_rf.py \
    --model_name oof_run1 \
    --grouping per_species \
    --train_on all \
    --feature_type both \
    --dataset_dir /gpfs/scratch/jvaska/CAMDA_AMR/AMR_v2/dnabert/inference/outputs/rf_datasets/oof/run1