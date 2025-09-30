#!/bin/bash
#SBATCH -p long-96core
#SBATCH -t 48:00:00
#SBATCH --output=res.txt

python -u infer_rf.py \
    --model_name train_new_test_new_flip \
    --grouping per_species \
    --testing_template /gpfs/scratch/jvaska/CAMDA_AMR/AMR_v2/data_pipeline/data/metadata/testing_template.csv \
    --models_dir /gpfs/scratch/jvaska/CAMDA_AMR/AMR_v2/rf/models/rf/per_species/train_new_test_new_flip \
    --test_dataset_dir /gpfs/scratch/jvaska/CAMDA_AMR/AMR_v2/dnabert/inference/outputs/rf_datasets/run2_testset/per_antibiotic/test \
    --out_path predictions/train_new_test_new_flip.csv \
    --feature_type hits