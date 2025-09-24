import pandas as pd
import os

"""#read full dataset
df = pd.read_csv('../data_pipeline/datasets/sequence_based/per_species/train/full_sequence_dataset.csv')

#find number of query ids that are unique to susceptible, unique to resistant, and shared between both
df_sus = df[df['phenotype'] == 1]
df_res = df[df['phenotype'] == 0]
unique_sus = set(df_sus['query_id'].unique())
unique_res = set(df_res['query_id'].unique())
shared = unique_sus.intersection(unique_res)
unique_sus_only = unique_sus - shared
unique_res_only = unique_res - shared
print(f'Unique to susceptible: {len(unique_sus_only)}')
print(f'Unique to resistant: {len(unique_res_only)}')
print(f'Shared: {len(shared)}')
print(f'Total: {len(unique_sus_only) + len(unique_res_only) + len(shared)}')
print(f'Total 2: {len(set(df_sus['query_id'].unique().tolist() + df_res['query_id'].unique().tolist()))}')"""

"""#read train dataset generated from pipeline v3 to ensure query_ids are balanced
df = pd.read_csv('../dnabert/finetune/data/per_antibiotic/finetune_augment_TESTSPLIT_v3/TET/train.csv')
sus_df = df[df['phenotype'] == 1]
res_df = df[df['phenotype'] == 0]
print(f'resistant queryid value counts: {res_df["query_id"].value_counts()}')
print(f'susceptible queryid value counts: {sus_df["query_id"].value_counts()}')"""

#predict mutual information between query_id and phenotype for final rf dataset to see if any dnabert features are useful

"""from sklearn.feature_selection import mutual_info_classif

#read final rf dataset
df = pd.read_csv('/gpfs/scratch/jvaska/CAMDA_AMR/AMR_v2/dnabert/inference/outputs/rf_datasets/per_antibiotic_models_v3/per_antibiotic/train/acinetobacter_baumannii/acinetobacter_baumannii_full_rf_dataset_test.csv')
X = df.drop(columns=['accession', 'ground_truth_phenotype'])
y = df['ground_truth_phenotype']

# X = your feature matrix (samples x features), y = labels (resistant/susceptible)
mi_scores = mutual_info_classif(X, y, discrete_features='auto')

# Put into a nice dataframe
mi_df = pd.DataFrame({
    "feature": X.columns,
    "mi_score": mi_scores
}).sort_values(by="mi_score", ascending=False)

print(mi_df.head(50))  # top 20 most informative features
print(mi_df[mi_df['feature'].str.contains('_hit_count')].head(50))"""

"""
# read eval results to determine best hyperparams
df = pd.read_csv('/gpfs/scratch/jvaska/CAMDA_AMR/AMR_v2/rf/eval_results/rf_models.csv')
print(df.tail())
print()
# get row where accuracy is max
print('overall best acc')
max_row = df.loc[df['accuracy'].idxmax()]
print(max_row)
print()

print('dnabert only best acc')
dnabert_only_df = df[df['feature_type'] == 'dnabert']
dnabert_only_max_row = dnabert_only_df.loc[dnabert_only_df['accuracy'].idxmax()]
print(dnabert_only_max_row)
print()

print('hits only best acc')
hits_only_df = df[df['feature_type'] == 'hits']
hits_only_max_row = hits_only_df.loc[hits_only_df['accuracy'].idxmax()]
print(hits_only_max_row)

"""


"""# get RF feature imporance from each species
import joblib
models_dir = '/gpfs/scratch/jvaska/CAMDA_AMR/AMR_v2/rf/models/rf/per_species/per_antibiotic_models_v3_both_dev_best_params'
for model_file in os.listdir(models_dir):
    if model_file.endswith('.joblib'):
        model_path = os.path.join(models_dir, model_file)
        model = joblib.load(model_path)
        print(model_path)
        df = pd.DataFrame(model.feature_importances_, index=model.feature_names_in_, columns=['importance'])
        #print top 100 feature types for each species
        df = df.sort_values(by='importance', ascending=False).head(100)
        #get number of pred_resistant and hit_count features
        num_pred_resistant = len([col for col in df.index if 'pred_resistant' in col])
        num_hit_count = len([col for col in df.index if 'hit_count' in col])
        print(f'Number of pred_resistant features: {num_pred_resistant}')
        print(f'Number of hit_count features: {num_hit_count}')
        print(f'%DNABERT features: {num_pred_resistant / (num_pred_resistant + num_hit_count) * 100}')
        print()
        """

"""df = pd.read_csv('/gpfs/scratch/jvaska/CAMDA_AMR/AMR_v2/dnabert/finetune/data/oof/run1/fold_1/meta_dataset.csv')
df['len'] = df['sequence'].str.len()
print(df['len'].value_counts())
print(len(df))"""
"""
fold_0_meta_accs = set(pd.read_csv('/gpfs/scratch/jvaska/CAMDA_AMR/AMR_v2/dnabert/finetune/data/oof/run1/fold_0/meta_dataset.csv')['accession'].tolist())
print(len(fold_0_meta_accs))

df_list = []

for species_dir in os.listdir('/gpfs/scratch/jvaska/CAMDA_AMR/AMR_v2/dnabert/inference/outputs/rf_datasets/run1_fold_0/full/train'):
    df_list.append(pd.read_csv(f'/gpfs/scratch/jvaska/CAMDA_AMR/AMR_v2/dnabert/inference/outputs/rf_datasets/run1_fold_0/full/train/{species_dir}/{species_dir}_full_rf_dataset.csv'))

fold_0_meta_train_set = pd.concat(df_list, axis=0)
print(fold_0_meta_train_set.head())
fold_0_meta_train_set_accs = set(fold_0_meta_train_set['accession'].tolist())

print(len(fold_0_meta_train_set_accs))

print(f'Intersection: {len(set(fold_0_meta_train_set_accs).intersection(set(fold_0_meta_accs)))}')"""
"""

#fold 0: how many meta_accs are missing from meta_dataset?
meta_accs = set([line for line in open('/gpfs/scratch/jvaska/CAMDA_AMR/AMR_v2/dnabert/finetune/data/oof/run1/fold_0/meta_accs_0.txt')])
meta_accs_in_dataset = set(pd.read_csv('/gpfs/scratch/jvaska/CAMDA_AMR/AMR_v2/dnabert/finetune/data/oof/run1/fold_0/meta_dataset.csv')['accession'].tolist())

print(f'Meta accs: {len(meta_accs)}')
print(f'Meta accs in dataset: {len(meta_accs_in_dataset)}')

missing_accs = meta_accs.difference(meta_accs_in_dataset)

#are the missing accs in the full dataset that we started with?
full_dataset_accs = set(pd.read_csv('/gpfs/scratch/jvaska/CAMDA_AMR/AMR_v2/data_pipeline/datasets/sequence_based/per_species/train/full_sequence_dataset.csv')['accession'].unique().tolist())
print(f'num mising accs: {len(missing_accs)}')
print(f'num missing accs in full set: {len(missing_accs.intersection(full_dataset_accs))}')"""

"""

df_list=[]
for species_dir in os.listdir('/gpfs/scratch/jvaska/CAMDA_AMR/AMR_v2/dnabert/inference/outputs/rf_datasets/run2_testset/per_antibiotic/test'):
    df_list.append(pd.read_csv(f'/gpfs/scratch/jvaska/CAMDA_AMR/AMR_v2/dnabert/inference/outputs/rf_datasets/run2_testset/per_antibiotic/test/{species_dir}/{species_dir}_full_rf_dataset.csv'))

full_test_df = pd.concat(df_list, axis=0)

accs_in_test_df = set(full_test_df['accession'].tolist())
print(f'accs in test df: {len(accs_in_test_df)}')

accs_in_test_template = set(pd.read_csv('/gpfs/scratch/jvaska/CAMDA_AMR/AMR_v2/data_pipeline/data/metadata/testing_template.csv')['accession'].tolist())
print(f'accs in test template: {len(accs_in_test_template)}')
print(f'intersection: {len(accs_in_test_df.intersection(accs_in_test_template))}')

#accs_in_full_test_dataset = set(pd.read_csv('/gpfs/scratch/jvaska/CAMDA_AMR/AMR_v2/data_pipeline/datasets/sequence_based/per_antibiotic/test/full_sequence_dataset.csv')['accession'].unique().tolist())
#print(f'accs in test dataset: {len(accs_in_full_test_dataset)}')

#print(f'intersection: {len(accs_in_full_test_dataset.intersection(accs_in_test_template))}')
"""



"""meta_dataset_df = pd.read_csv('/gpfs/scratch/jvaska/CAMDA_AMR/AMR_v2/dnabert/finetune/data/oof/run2/fold_2/meta_dataset.csv')
print(meta_dataset_df['sequence'].isnull().sum())

bad_rows = meta_dataset_df[~meta_dataset_df['sequence'].apply(lambda x: isinstance(x, str))]
print(bad_rows.head())"""


"""
models_dir = '/gpfs/scratch/jvaska/CAMDA_AMR/AMR_v2/rf/models/rf/per_species/oof_run2_allfolds_HITSONLY'
for model_file in os.listdir(models_dir):
    if model_file.endswith('.joblib'):
        model_path = os.path.join(models_dir, model_file)
        model = joblib.load(model_path)
        print(model_path)
        df = pd.DataFrame(model.feature_importances_, index=model.feature_names_in_, columns=['importance'])
        print(len(df))
        print(df.head(50))"""

import joblib
model_path = '/gpfs/scratch/jvaska/CAMDA_AMR/AMR_v2/rf/models/xgb/per_species/oof_run2_01_both_xgb/neisseria_gonorrhoeae_xgb_model.joblib'
model = joblib.load(model_path)
print(model_path)
features = list(model.feature_names_in_)
print(features)
