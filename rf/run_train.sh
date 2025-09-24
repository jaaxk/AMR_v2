#!/bin/bash
#SBATCH -p long-96core
#SBATCH -t 48:00:00
#SBATCH --output=res.txt

python -u train_rf.py \
    --model_name run2_feature_plot_oof_stack \
    --grouping per_species \
    --train_on 01 \
    --feature_type both \
    --dataset_dir /gpfs/scratch/jvaska/CAMDA_AMR/AMR_v2/dnabert/inference/outputs/rf_datasets/run2_trainset \
    --model_type xgb \
    --oof_stack \
    --interleave \
    --feature_plot /gpfs/scratch/jvaska/CAMDA_AMR/AMR_v2/rf/models/oof/run2_hitsonly_OOF/hits/exclude_fold_2 \


#    --feature_plot /gpfs/scratch/jvaska/CAMDA_AMR/AMR_v2/rf/models/rf/per_species/oof_run3_01_both_rf
    
#    --feature_plot /gpfs/scratch/jvaska/CAMDA_AMR/AMR_v2/rf/models/xgb/per_species/oof_run2_01_both_xgb
