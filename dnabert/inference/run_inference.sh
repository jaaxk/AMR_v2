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
        --dataset_dir /gpfs/scratch/jvaska/CAMDA_AMR/AMR_v2/data_pipeline/datasets/sequence_based/per_species/train \
        --model_path /gpfs/scratch/jvaska/CAMDA_AMR/AMR_v2/dnabert/finetune/finetuned_models/full_model_species_only_v1/best \
        --model_name full_model_species_only_v1 \
        --grouping per_species \
        --train True