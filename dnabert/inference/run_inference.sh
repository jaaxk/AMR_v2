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
        --dataset_dir /gpfs/scratch/jvaska/CAMDA_AMR/AMR_v2/dnabert/finetune/data/oof/run1/fold_2/TEST_DELETE.csv \
        --model_path /gpfs/scratch/jvaska/CAMDA_AMR/AMR_v2/dnabert/finetune/finetuned_models/oof_run1/fold_0/best \
        --run_name test \
        --grouping full \
	--return_logits True \
	--train True
