#!/bin/bash
#SBATCH -p a100-long
#SBATCH --gres=gpu:1
#SBATCH -t 48:00:00
#SBATCH --output=res1.txt

python -u train_rf.py \
    --model_name run6_dnabert_alllogits_nn \
    --grouping per_species \
    --train_on 01 \
    --feature_type both \
    --dataset_dir /gpfs/scratch/jvaska/CAMDA_AMR/AMR_v2/dnabert/inference/outputs/rf_datasets/run6_alllogits \
    --model_type nn \
    --eval

# place script in model output directory, if it exists
MODEL_OUTPUT_DIR=models/${model_type}/${grouping}/${model_name}
if [ -d "${MODEL_OUTPUT_DIR}" ]; then
    cp $0 ${MODEL_OUTPUT_DIR}
fi

