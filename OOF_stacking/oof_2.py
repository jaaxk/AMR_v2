"""
Run inference on each fold using finetuned model that did not see that fold's data
Save all logits and train one RF with all of them
Should be run on GPU (a100) node
"""


# imports
import argparse
import pandas as pd
import os
import subprocess


#lists
species_list = [
    'neisseria_gonorrhoeae', 
    'staphylococcus_aureus', 
    'streptococcus_pneumoniae', 
    'salmonella_enterica', 
    'klebsiella_pneumoniae', 
    'escherichia_coli', 
    'pseudomonas_aeruginosa', 
    'acinetobacter_baumannii', 
    'campylobacter_jejuni' 
]



# command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--data_base_dir", type=str, help='for training, pass base directory containing all folds, for inference pass full test dataset (not dir)', default="/gpfs/scratch/jvaska/CAMDA_AMR/AMR_v2/dnabert/finetune/data/oof/run1")
parser.add_argument("--models_base_dir", type=str, default="/gpfs/scratch/jvaska/CAMDA_AMR/AMR_v2/dnabert/finetune/finetuned_models/oof_run1")
parser.add_argument("--run_name", type=str, default="run1")
parser.add_argument("--train", type=bool, default=False)
parser.add_argument("--grouping", choices=["full", "per_species", "per_antibiotic"], default="full")
parser.add_argument("--base_dir", type=str, default="/gpfs/scratch/jvaska/CAMDA_AMR/AMR_v2", help="Base directory of entire project (AMR_v2)")
args = parser.parse_args()
train = 'True' if args.train else 'False'

if args.train: #if were on the train set, run inference on opposite model
    print('Running in TRAIN mode')
    for fold in os.listdir(args.data_base_dir):
        print(f"Running inference on fold {fold}")
        
        data_dir = os.path.join(args.data_base_dir, fold)
        model_dir = os.path.join(args.models_base_dir, fold)

        cmd = f"python {args.base_dir}/dnabert/inference/inference.py --model_path {model_dir}/best --dataset_dir {data_dir}/meta_dataset.csv \
            --output_format random_forest --run_name {args.run_name}_{fold} --grouping {args.grouping} --train {train} --return_logits True"
        subprocess.run(cmd, shell=True)

    # collect the separate datasets we created and combine
    if args.grouping =='full':
        rf_datasets_base_dir = f'{args.base_dir}/dnabert/inference/outputs/rf_datasets/{args.run_name}'
        final_rf_out_dir = f'{args.base_dir}/dnabert/inference/outputs/rf_datasets/oof/{args.run_name}'
        for species in species_list:
            final_df = pd.DataFrame()
            for fold in os.listdir(args.data_base_dir):
                fold_df_path = f'{rf_datasets_base_dir}_{fold}/full/train/{species}/{species}_full_rf_dataset.csv'
                fold_df = pd.read_csv(fold_df_path)
                final_df = pd.concat([final_df, fold_df])

            os.makedirs(f'{final_rf_out_dir}/{species}', exist_ok=True)
            final_df.to_csv(f'{final_rf_out_dir}/{species}/{species}_full_rf_dataset.csv', index=False)





    elif args.grouping =='per_species':
        raise NotImplementedError("Not implemented for per_species grouping")
    elif args.grouping =='per_antibiotic':
        raise NotImplementedError("Not implemented for per_antibiotic grouping")

    print(f'Inference complete, now training RF models on the datasets here: {final_rf_out_dir}...')

    cmd = f'python {args.base_dir}/rf/train_rf.py \
    --model_name {args.run_name} \
    --grouping {args.grouping} \
    --train_on all \
    --feature_type both \
    --dataset_dir {final_rf_out_dir}'
    # no held out test set, meant for testing on independent validation set. We can split this into train/test in the future for evaluation








else: #if final inference, we can run inference on each model, then average logits
    print('Running in TEST mode')

    if args.grouping == 'full':
        for model in os.listdir(args.models_base_dir):
            cmd = f"python {args.base_dir}/dnabert/inference/inference.py --run_name {args.run_name}_{model} --model_path {args.models_base_dir}/{model}/best --dataset_dir {args.data_base_dir} \
            --output_format random_forest --grouping full --return_logits True"

            subprocess.run(cmd, shell=True)

    elif args.grouping == 'per_species':
        raise NotImplementedError("Not implemented for per_species grouping")
    elif args.grouping == 'per_antibiotic':
        raise NotImplementedError("Not implemented for per_antibiotic grouping")

    # open datasets and average logits
    for species in species_list:
        dfs = []
        for model in os.listdir(args.models_base_dir):

            rf_dataset_path = f'{args.base_dir}/dnabert/inference/outputs/rf_datasets/{args.run_name}_{model}/full/test/{species}/{species}_full_rf_dataset.csv'
            df = pd.read_csv(rf_dataset_path)
            dfs.append(df)
        
        pred_cols = [col for col in dfs[0].columns if col.endswith('_pred_resistant')]
        avg_preds = sum([df[pred_cols] for df in dfs]) / len(dfs)
        static_cols = dfs[0].drop(columns=pred_cols)


        final_df = pd.concat([static_cols, avg_preds], axis=1)
        os.makedirs(f'{args.base_dir}/dnabert/inference/outputs/rf_datasets/oof/{args.run_name}_averaged/{species}', exist_ok=True)
        final_df.to_csv(f'{args.base_dir}/dnabert/inference/outputs/rf_datasets/oof/{args.run_name}_averaged/{species}/{species}_full_rf_dataset.csv', index=False)
        print(f'Done, saved final TEST set to: {args.base_dir}/dnabert/inference/outputs/rf_datasets/oof/{args.run_name}_averaged/{species}/{species}_full_rf_dataset.csv')

    

