#!/bin/bash
#SBATCH -p long-40core
#SBATCH -t 48:00:00
#SBATCH --output=res.txt
#SBATCH --mem 150G

python -u oof_1.py \
	--run_name run2 \
	--grouping per_antibiotic
