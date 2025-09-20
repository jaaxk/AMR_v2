#!/bin/bash
#SBATCH -p long-96core
#SBATCH -t 48:00:00
#SBATCH --output=res.txt

python -u infer_rf.py \
    --model_name test_dev_v3_dnabertonly \
    --grouping per_species \
    --testing_template /gpfs/scratch/jvaska/CAMDA_AMR/AMR_v2/data_pipeline/data/metadata/testing_template.csv \
    --models_dir /gpfs/scratch/jvaska/CAMDA_AMR/AMR_v2/rf/models/rf/per_species/test_dev_v3_dnabertonly \
    --test_dataset_dir /gpfs/scratch/jvaska/CAMDA_AMR/AMR_v2/dnabert/inference/outputs/rf_datasets/per_antibiotic_models_v3/per_antibiotic/test 
