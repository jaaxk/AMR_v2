#!/bin/bash
#SBATCH -p a100-long
#SBATCH --output res.txt
#SBATCH --gres=gpu:1
#SBATCH --time 48:00:00
#SBATCH --mem 120G

python -u oof_2.py \
    --data_base_dir /gpfs/scratch/jvaska/CAMDA_AMR/AMR_v2/dnabert/finetune/data/oof/run6_1000bp \
    --models_base_dir /gpfs/scratch/jvaska/CAMDA_AMR/AMR_v2/dnabert/finetune/finetuned_models/run6_1000bp \
    --run_name run6_1000bp_alllogits \
    --grouping full \
    --train \
    --return_logits all

