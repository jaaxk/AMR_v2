#!/bin/bash

conda activate dna

python inference.py \
        --output_format random_forest \
        --dataset_dir /gpfs/scratch/jvaska/CAMDA_AMR/AMR_v2/data_pipeline/datasets/sequence_based/per_species/train \
        --model_path 
