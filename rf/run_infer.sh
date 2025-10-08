#!/bin/bash
#SBATCH -p long-96core
#SBATCH -t 48:00:00
#SBATCH --output=res.txt

python -u infer_rf.py \
    --model_name run6_1000bp_avglogits_oofstack \
    --grouping per_species \
    --testing_template /gpfs/scratch/jvaska/CAMDA_AMR/AMR_v2/data_pipeline/data/metadata/testing_template.csv \
    --models_dir /gpfs/scratch/jvaska/CAMDA_AMR/AMR_v2/rf/models/oof/run6_1000bp_dnabert_avglogits_oofstack_all \
    --test_dataset_dir /gpfs/scratch/jvaska/CAMDA_AMR/AMR_v2/dnabert/inference/outputs/rf_datasets/run6_1000bp_avglogits_testset_testset/full/test \
    --out_path predictions/run6_1000bp_avglogits_oofstack.csv \
    --feature_type both \
    --oof_stack \
    --top_15pct_features /gpfs/scratch/jvaska/CAMDA_AMR/AMR_v2/data_analysis/top_15p_features
