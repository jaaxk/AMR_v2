#!/bin/bash
#SBATCH -p hbm-long-96core
#SBATCH -t 48:00:00
#SBATCH --output=res.txt

python -u train_rf.py \
    --model_name run4_oofstack_01_rf \
    --grouping per_species \
    --train_on 01 \
    --feature_type both \
    --dataset_dir /gpfs/scratch/jvaska/CAMDA_AMR/AMR_v2/dnabert/inference/outputs/rf_datasets/run2_trainset \
    --model_type rf \
    --flip_phenotype /gpfs/scratch/jvaska/CAMDA_AMR/AMR_v2/data_analysis/mismatches \
    --feature_plot /gpfs/scratch/jvaska/CAMDA_AMR/AMR_v2/rf/models/rf/per_species/run4_all_both_rf \
    --oof_stack

# place script in model output directory, if it exists
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
MODEL_OUTPUT_DIR=models/${model_type}/${grouping}/${model_name}
if [ -d "${MODEL_OUTPUT_DIR}" ]; then
    cp ${SCRIPT_DIR}/run_infer.sh ${MODEL_OUTPUT_DIR}
fi

