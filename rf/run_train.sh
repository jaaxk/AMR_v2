#!/bin/bash
#SBATCH -p hbm-long-96core
#SBATCH -t 48:00:00
#SBATCH --output=res1.txt

python -u train_rf.py \
    --model_name run5_1_featureplot_1to200 \
    --grouping per_species \
    --train_on 01 \
    --feature_type both \
    --dataset_dir /gpfs/scratch/jvaska/CAMDA_AMR/AMR_v2/dnabert/inference/outputs/rf_datasets/run5_1_trainset \
    --model_type rf \
    --feature_plot /gpfs/scratch/jvaska/CAMDA_AMR/AMR_v2/rf/models/rf/per_species/run5_1_rf_all_top15dnabert_numpredres
    
# place script in model output directory, if it exists
MODEL_OUTPUT_DIR=models/${model_type}/${grouping}/${model_name}
if [ -d "${MODEL_OUTPUT_DIR}" ]; then
    cp $0 ${MODEL_OUTPUT_DIR}
fi

