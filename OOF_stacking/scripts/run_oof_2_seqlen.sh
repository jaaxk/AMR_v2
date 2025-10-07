#!/bin/bash
#SBATCH -p a100-long
#SBATCH --output res1.txt
#SBATCH --gres=gpu:4
#SBATCH --time 48:00:00
#SBATCH --mem 240G

for seq_len in 1000 1500 2000 2500; do
    echo "Running for seq_len ${seq_len}"

    python -u oof_2.py \
        --data_base_dir /gpfs/scratch/jvaska/CAMDA_AMR/AMR_v2/dnabert/finetune/data/oof/run6_${seq_len}bp \
        --models_base_dir /gpfs/scratch/jvaska/CAMDA_AMR/AMR_v2/dnabert/finetune/finetuned_models/run6_${seq_len}bp \
        --run_name run6_${seq_len}bp \
        --grouping full \
        --train

    done
