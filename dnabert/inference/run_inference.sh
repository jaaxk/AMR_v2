#!/bin/bash
#SBATCH -p a100-long
#SBATCH -t 24:00:00
#SBATCH --output=res.txt
#SBATCH --gres=gpu:1
#SBATCH --mem 120G

#conda activate dna

python -u inference.py \
        --output_format consensus \
        --dataset_dir /gpfs/scratch/jvaska/CAMDA_AMR/AMR_v2/data_pipeline/datasets/run6_1000bp/sequence_based/per_antibiotic/train/full_sequence_dataset.csv \
        --model_path /gpfs/scratch/jvaska/CAMDA_AMR/AMR_v2/dnabert/finetune/finetuned_models/run6_1000bp/fold_2/best \
        --run_name run6_1000bp_alllogits_fold_2 \
        --grouping full \
        --train True
