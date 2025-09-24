#!/bin/bash
#SBATCH -p long-96core
#SBATCH -t 48:00:00
#SBATCH --output=res.txt

python -u infer_rf.py \
    --model_name OOF_run2_allfolds_hitsonly_nomi_defaultparams \
    --grouping per_species \
    --testing_template /gpfs/scratch/jvaska/CAMDA_AMR/AMR_v2/data_pipeline/data/metadata/testing_template.csv \
    --models_dir /gpfs/scratch/jvaska/CAMDA_AMR/AMR_v2/rf/models/rf/per_species/oof_run2_allfolds_hitsonly_nomi_defaultparams \
    --test_dataset_dir /gpfs/scratch/jvaska/CAMDA_AMR/AMR_v2/dnabert/inference/outputs/rf_datasets/run2_testset/per_antibiotic/test \
    --out_path predictions/oof_run2_allfolds_hitsonly_nomi_defaultparams \
    --feature_type hits