#!/bin/bash
#SBATCH -p hbm-long-96core
#SBATCH -t 48:00:00
#SBATCH --output=res1.txt

python -u train_rf.py \
    --model_name run6_1000bp_dnabert_avglogits_oofstack_all \
    --grouping per_species \
    --train_on all \
    --feature_type both \
    --dataset_dir /gpfs/scratch/jvaska/CAMDA_AMR/AMR_v2/dnabert/inference/outputs/rf_datasets/run6_1000bp_avglogits \
    --top_15pct_features /gpfs/scratch/jvaska/CAMDA_AMR/AMR_v2/data_analysis/top_15p_features \
    --model_type rf \
    --oof_stack

# place script in model output directory, if it exists
MODEL_OUTPUT_DIR=models/${model_type}/${grouping}/${model_name}
if [ -d "${MODEL_OUTPUT_DIR}" ]; then
    cp $0 ${MODEL_OUTPUT_DIR}
fi

