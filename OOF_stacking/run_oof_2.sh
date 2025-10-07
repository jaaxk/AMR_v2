#!/bin/bash
#SBATCH -p a100-long
#SBATCH --output res.txt
#SBATCH --gres gpu:1
#SBATCH --time 48:00:00
#SBATCH --mem 250G

python -u oof_2.py \
    --data_base_dir /gpfs/scratch/jvaska/CAMDA_AMR/AMR_v2/data_pipeline/datasets/sequence_based/per_antibiotic/test/full_sequence_dataset.csv \
    --models_base_dir /gpfs/scratch/jvaska/CAMDA_AMR/AMR_v2/dnabert/finetune/finetuned_models/run6_1000bp/fold_FULL/best \
    --run_name run6_1000bp_avglogits_testset \
    --eval_on_full_models \
    --grouping full \
    --return_logits average

