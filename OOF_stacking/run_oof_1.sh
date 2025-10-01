#!/bin/bash
#SBATCH -p a100-long
#SBATCH -t 48:00:00
#SBATCH --gres=gpu:4
#SBATCH --mem 250G
#SBATCH --output logs/%j.txt

for bp in 1000 1500 2000 2500; do
	python -u oof_1.py \
		--run_name run6_${bp}bp \
		--grouping full \
		--full_sequence_dataset_path /gpfs/scratch/jvaska/CAMDA_AMR/AMR_v2/data_pipeline/datasets/run6_${bp}bp/sequence_based/per_antibiotic/train/full_sequence_dataset.csv
	done
