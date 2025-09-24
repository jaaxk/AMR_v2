"""
Perform Out-Of-Fold (OOF) stacking for DNABERT and RF or XGBoost meta models

Steps:
 - Break entire training set into k folds
    - The higher k, the more data the model has to train on, but the more computationally expensive (training more DNABERT models)
    - We can start with k=4 for now
 - For each fold: 
    - Break into 90/10 train and val for DNABERT (make dev and test the same)
    - Train DNABERT on all other folds (k-1)
    - Use DNABERT to predict on fold and save predictions (logits)
 - Train RF or XGBoost on combined logits from all folds (so all predictions came from a model that never saw the data)


For k=4:
 - 5500 samples/3 = 1833 samples per fold
 - 1833 samples * 2 = 3666 samples
 - 3666 samples * .9 = 3299 samples for DNABERT training
 - 3666 samples * .1 = 366 samples for DNABERT validation
 - we will use full models for now to save compute (only train 4 models)
    - if using per-antibiotic models, 3960/4 = 990 samples per antibiotic model (compared to 1100 before OOF, ~10% less data)
    - this would be training 16 different models

"""

# imports
import argparse
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import shutil
import os
import subprocess

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--k", type=int, default=3)
  parser.add_argument("--train_metadata_path", type=str, default="/gpfs/scratch/jvaska/CAMDA_AMR/AMR_v2/data_pipeline/data/metadata/training_dataset.csv")
  parser.add_argument('--full_sequence_dataset_path', type=str, default="/gpfs/scratch/jvaska/CAMDA_AMR/AMR_v2/data_pipeline/datasets/sequence_based/per_antibiotic/train/full_sequence_dataset.csv")
  parser.add_argument('--run_name', type=str, default="run1")
  parser.add_argument('--grouping', choices=['full', 'per_species', 'per_antibiotic'], default='per_antibiotic')
  args = parser.parse_args()
  


  # load all accessions in CAMDA train set
  all_accessions = list(set(pd.read_csv(args.train_metadata_path)['accession'].tolist()))

  # break into k folds
  kf = KFold(n_splits=args.k, shuffle=True, random_state=42)
  folds = list(kf.split(all_accessions))
  folds.append(([i for i in range(len(all_accessions))], []))

  # setup output directory and lists
  dnabert_accs_so_far = []
  base_out_dir = f"/gpfs/scratch/jvaska/CAMDA_AMR/AMR_v2/dnabert/finetune/data/oof/{args.run_name}"
  os.makedirs(base_out_dir, exist_ok=True)
  
  # main loop
  for i, (dnabert_indices, meta_indices) in enumerate(folds):
    if i == args.k:
      print('Starting FULL DATASET fold')
      fold_out_dir = f"{base_out_dir}/fold_FULL"
    else:
      print(f"Starting fold {i+1}")
      fold_out_dir = f"{base_out_dir}/fold_{i}"
    os.makedirs(fold_out_dir, exist_ok=True)
    """
    
      
    # get train and val accessions for DNABERT
    #split fold[0] into 90/10 train/val
    train_indices, dev_indices = train_test_split(dnabert_indices, test_size=0.1, random_state=42)
    train_accessions = [all_accessions[i] for i in train_indices]
    dev_accessions = [all_accessions[i] for i in dev_indices]
    #write to txt
    train_accs_path = f"{fold_out_dir}/train_accessions.txt"
    dev_accs_path = f"{fold_out_dir}/dev_accessions.txt"
    test_accs_path = f"{fold_out_dir}/test_accessions.txt"
    with open(train_accs_path, 'w') as f:
      for acc in train_accessions:
        f.write(f"{acc}\n")
    with open(dev_accs_path, 'w') as f:
      for acc in dev_accessions:
        f.write(f"{acc}\n")
    with open(test_accs_path, 'w') as f: #test is the same as dev
      for acc in dev_accessions:
        f.write(f"{acc}\n")

    # write dnabert and meta accessions to fold path
    dnabert_accs = [all_accessions[i] for i in dnabert_indices]
    meta_accs = [all_accessions[i] for i in meta_indices]
    with open(f"{fold_out_dir}/dnabert_accs_{i}.txt", 'w') as f:
      for acc in dnabert_accs:
        f.write(f"{acc}\n")
    with open(f"{fold_out_dir}/meta_accs_{i}.txt", 'w') as f:
      for acc in meta_accs:
        f.write(f"{acc}\n")

    # ensure no overlap across folds
    if train_accessions in dnabert_accs_so_far or dev_accessions in dnabert_accs_so_far:
      raise ValueError("Overlap detected between folds")
    dnabert_accs_so_far.extend(train_accessions)
    dnabert_accs_so_far.extend(dev_accessions)
    
    # create data split with our dnabert_preprocessing.py script
    preprocessing_script_path = '../dnabert/finetune/data/dnabert_preprocessing.py' #relative path!!
    cmd = f"python {preprocessing_script_path} --balance_method stratify --out_dir {fold_out_dir}/dnabert_data \
        --full_dataset_path {args.full_sequence_dataset_path} --dev_accs_path {dev_accs_path} --train_accs_path {train_accs_path} \
        --test_accs_path {test_accs_path} --grouping {args.grouping}" #using full models for now to save compute
    print(cmd)
    # run command
    proc = subprocess.run(
        cmd,
        shell=True,
        capture_output=True,
        text=True
    )

    if proc.returncode == 0:
        print("Preprocessing finished successfully ✅")
    else:
        print(f"Preprocessing failed with exit code {proc.returncode} ❌")
        print("---- STDOUT ----")
        print(proc.stdout)
        print("---- STDERR ----")
        print(proc.stderr)

    # get meta dataset
  """
    meta_accs_path = f"{fold_out_dir}/meta_accs_{i}.txt"
    meta_accs = [line.strip() for line in open(meta_accs_path)]
    meta_df = pd.read_csv(args.full_sequence_dataset_path)
    meta_df = meta_df[meta_df['accession'].isin(meta_accs)]
    meta_df.to_csv(f"{fold_out_dir}/meta_dataset.csv", index=False)


    # 

  """
  # finetune DNABERT on the data in fold_out_dir
  # run finetune.py script
  if args.grouping == 'full':
    finetune_script_path = '../dnabert/finetune/finetune_scripts/run_finetune_oof.sh' #relative path!!
  elif args.grouping == 'per_antibiotic':
    finetune_script_path = '../dnabert/finetune/finetune_scripts/run_finetune_oof_perantibiotic.sh'
  cmd = f"sbatch {finetune_script_path} {base_out_dir} {args.run_name}" #using full models for now to save compute
  # run command
  print('RUN this command if didnt automatically run:')
  print(cmd)
  subprocess.call(cmd, shell=True)
  print()
  print('Next: wait for finetuning to finish and then run oof_2.py')





    
      
"""

if __name__ == "__main__":
    main()
        
        