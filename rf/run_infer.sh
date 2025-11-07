#!/bin/bash
#SBATCH -p long-96core
#SBATCH -t 48:00:00
#SBATCH --output=res.txt

python -u infer_rf.py \
    --model_name run6_cb_numpredres_bestparams \
    --grouping per_species \
    --testing_template /gpfs/scratch/jvaska/CAMDA_AMR/AMR_v2/data_pipeline/data/metadata/testing_template.csv \
    --models_dir /gpfs/scratch/jvaska/CAMDA_AMR/AMR_v2/rf/models/cb/per_species/best_hyperparams_cb_all \
    --test_dataset_dir /gpfs/scratch/jvaska/CAMDA_AMR/AMR_v2/dnabert/inference/outputs/rf_datasets/run3_testset/per_antibiotic/test \
    --out_path predictions/run6_cb_numpredres_bestparams.csv \
    --feature_type both \
    --top_15pct_features /gpfs/scratch/jvaska/CAMDA_AMR/AMR_v2/data_analysis/top_15p_features \
    --model_type cb
