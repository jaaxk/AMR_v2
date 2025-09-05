#!/bin/bash
#SBATCH -p hbm-long-96core
#SBATCH -t 48:00:00
#SBATCH --output=res.txt

python pipeline.py \
	--assemblies_dir /gpfs/scratch/jvaska/CAMDA_AMR/AMR_v2/data_pipeline/data/assemblies/train \
	--metadata_path /gpfs/scratch/jvaska/CAMDA_AMR/AMR_v2/data_pipeline/data/metadata/training_dataset.csv \
	--model_type sequence_based \
	--grouping per_species \
	--train True \
	--perspecies_dbgwas_dir ./data/dbgwas/p0.05/per_species

