import pandas as pd

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



