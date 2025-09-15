#!/bin/bash
#SBATCH -p a100-long
#SBATCH -t 24:00:00
#SBATCH --output=res.txt
#SBATCH --gres=gpu:1
#SBATCH --mem 250G

conda activate dna

#change model_path once we have a trained full model
python -u inference.py \
        --output_format random_forest \
        --dataset_dir /gpfs/scratch/jvaska/CAMDA_AMR/AMR_v2/data_pipeline/datasets/sequence_based/per_antibiotic/test \
        --model_path /gpfs/scratch/jvaska/CAMDA_AMR/AMR_v2/dnabert/finetune/finetuned_models/per_antibiotic_models_v3 \
        --model_name per_antibiotic_models_v3 \
        --grouping per_antibiotic \
        --split True
