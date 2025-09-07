"""
SPLITS the training dataset into train/test/dev set according to dev_accs.txt
BALANCES dataset by subsampling rows where phenotype=0 (resistant) by ~1/3 *this should be changed with other datasets that have different class imbalances
SUBSAMPLES train set down to ~500k rows for memory issues
dev.csv and test.csv are the exact same (to increase training data)
evaluation should be done on independent validation set on CAMDA website TODO: make this script also support a separate test_accs.csv to make fully independent train/test/dev sets for evaluation without independent test set (if CAMDA site is no longer supported)

"""

FULL_DATASET_PATH="../../../data_pipeline/datasets/sequence_based/per_species/train/full_sequence_dataset.csv"
DEV_ACCS_PATH="dev_accs.txt"
OUT_DIR='./dnabert_finetune_dataset_v1'

import os
import pandas as pd

df = pd.read_csv(FULL_DATASET_PATH)
dev_accs = [line.rstrip() for line in open(DEV_ACCS_PATH)]
print(f'full dataset size: {len(df)}')

dev_df = df[df['accession'].isin(dev_accs)]
print(f'dev dataset size: {len(dev_df)}')
dev_df.to_csv(os.path.join(OUT_DIR, 'dev.csv'), index=False)
dev_df.to_csv(os.path.join(OUT_DIR, 'test.csv'), index=False) #dev and test are the same

train_df = df[~df['accession'].isin(dev_accs)]
print(f'train dataset size: {len(train_df)}')

#we need to (1) subsample the train set to ~500k rows for memory isssues and (2) balance the dataset by removing mostly rows where 'phenotype'=0 (resistant)
# 0=Resistant, 1=Suscptible
print(f'train dataset phenotype balance: {train_df["phenotype"].value_counts()}')
#original split is ~1,500,000 : ~500,000 resistant:suscptible. Need to reduce resitant by ~1/3
train_df = pd.concat([train_df[train_df['phenotype'] == 0].sample(frac=0.33, random_state=42), train_df[train_df['phenotype'] == 1]])
train_df = train_df.sample(frac=1, random_state=42) #shuffle
print(f'train dataset size after balancing: {len(train_df)}')
train_df = train_df.sample(n=500000, random_state=42) #memory error over 500k rows with 4GPUS, DDP, 250G memory
print(f'train dataset phenotype balance after balancing: {train_df["phenotype"].value_counts()}')


train_df.to_csv(os.path.join(OUT_DIR, 'train.csv'), index=False)