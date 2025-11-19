import pandas as pd
import os

species_mapping = {
    'klebsiella_pneumoniae': 0,
    'streptococcus_pneumoniae': 1,
    'escherichia_coli': 2,
    'campylobacter_jejuni': 3,
    'salmonella_enterica': 4,
    'neisseria_gonorrhoeae': 5,
    'staphylococcus_aureus': 6,
    'pseudomonas_aeruginosa': 7,
    'acinetobacter_baumannii': 8
}

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

"""import joblib
model_path = '/gpfs/scratch/jvaska/CAMDA_AMR/AMR_v2/rf/models/xgb/per_species/oof_run2_01_both_xgb/neisseria_gonorrhoeae_xgb_model.joblib'
model = joblib.load(model_path)
print(model_path)
features = list(model.feature_names_in_)
print(features)"""

"""

import os
from pathlib import Path
import pandas as pd
import numpy as np

# Species mapping for iteration


# Paths provided
NEW_TRAIN_ROOT = Path("/gpfs/scratch/jvaska/CAMDA_AMR/AMR_v2/dnabert/inference/outputs/rf_datasets/oof/run2")
OLD_TRAIN_FULL_ROOT = Path("/gpfs/scratch/jvaska/CAMDA_AMR/AMR_v2/dnabert/inference/outputs/rf_datasets/original_dataset/train/FULL")

NEW_TEST_ROOT = Path("/gpfs/scratch/jvaska/CAMDA_AMR/AMR_v2/dnabert/inference/outputs/rf_datasets/run2_testset")
OLD_TEST_ROOT = Path("/gpfs/scratch/jvaska/CAMDA_AMR/AMR_v2/dnabert/inference/outputs/rf_datasets/original_dataset/test")

def find_first_existing(paths):
    for p in paths:
        if p.exists():
            return p
    return None

def candidate_new_train_paths(species):
    # Handle possible filename typo (_datase.csv vs _dataset.csv)
    return [
        NEW_TRAIN_ROOT / species / f"{species}_full_rf_dataset.csv",
        NEW_TRAIN_ROOT / species / f"{species}_full_rf_datase.csv",
    ]

def candidate_new_test_paths(species):
    # We don't know exact nesting; search a few plausible patterns and fall back to glob
    candidates = [
        NEW_TEST_ROOT / "per_antibiotic" / "test" / species / f"{species}_full_rf_dataset.csv",
        NEW_TEST_ROOT / species / f"{species}_full_rf_dataset.csv",
        NEW_TEST_ROOT / "test" / species / f"{species}_full_rf_dataset.csv",
    ]
    existing = find_first_existing(candidates)
    if existing:
        return existing
    # fallback: glob
    hits = list(NEW_TEST_ROOT.rglob(f"{species}_full_rf_dataset.csv"))
    return hits[0] if hits else None

def load_df(p: Path, kind: str, species: str):
    if p is None:
        print(f"[WARN] {kind} dataset missing for {species}")
        return None
    try:
        return pd.read_csv(p)
    except Exception as e:
        print(f"[ERROR] Failed reading {kind} for {species} at {p}: {e}")
        return None

def filter_hits_only_train(df: pd.DataFrame):
    # Mirror train_rf.filter_feature_type for 'hits'
    df = df.copy()
    meta_cols = [c for c in ['accession','species','antibiotic','ground_truth_phenotype'] if c in df.columns]
    hits_cols = [c for c in df.columns if 'hit_count' in c]
    return df[meta_cols + hits_cols]

def filter_hits_only_infer(df: pd.DataFrame):
    # Mirror infer_rf.filter_feature_type for 'hits'
    df = df.copy()
    # species/antibiotic may not exist
    for c in ['species','antibiotic']:
        if c in df.columns:
            df = df.drop(columns=c)
    # keep accession if present
    cols = [c for c in df.columns if 'hit_count' in c]
    if 'accession' in df.columns:
        cols = ['accession'] + cols
    return df[cols]

def compare_feature_sets(hits_old, hits_new):
    set_old = set(hits_old)
    set_new = set(hits_new)
    only_old = sorted(list(set_old - set_new))
    only_new = sorted(list(set_new - set_old))
    same_order = (hits_old == hits_new)
    return only_old, only_new, same_order

def per_feature_numeric_diffs(df_old, df_new, feature_cols):
    # Compare on the intersection of accessions if present; else compare row-aligned
    if 'accession' in df_old.columns and 'accession' in df_new.columns:
        common = sorted(list(set(df_old['accession']) & set(df_new['accession'])))
        A = df_old.set_index('accession').loc[common, feature_cols]
        B = df_new.set_index('accession').loc[common, feature_cols]
    else:
        # align by index length minimum
        n = min(len(df_old), len(df_new))
        A = df_old.iloc[:n][feature_cols]
        B = df_new.iloc[:n][feature_cols]

    diffs = (A - B).abs()
    summary = pd.DataFrame({
        'mean_abs_diff': diffs.mean(axis=0),
        'nonzero_diff_count': (diffs > 0).sum(axis=0)
    }).sort_values(by=['nonzero_diff_count','mean_abs_diff'], ascending=False)
    return summary, len(A)

def summarize_metadata(df, is_train):
    out = {}
    if 'species' in df.columns:
        out['species_unique'] = sorted(df['species'].unique().tolist())
    if 'antibiotic' in df.columns:
        out['antibiotic_unique'] = sorted(pd.Series(df['antibiotic']).unique().tolist())
    if is_train and 'ground_truth_phenotype' in df.columns:
        out['phenotype_counts'] = df['ground_truth_phenotype'].value_counts().to_dict()
    if 'accession' in df.columns:
        out['n_accessions'] = df['accession'].nunique()
    return out

def analyze_pair(old_df, new_df, species, is_train):
    label = "TRAIN" if is_train else "TEST"
    if old_df is None or new_df is None:
        print(f"[SKIP] {label} {species}: one of the datasets is missing.")
        return

    # Mirror the filtering as close as possible to model input (hits-only)
    old_f = filter_hits_only_train(old_df) if is_train else filter_hits_only_infer(old_df)
    new_f = filter_hits_only_train(new_df) if is_train else filter_hits_only_infer(new_df)

    # Accessions overlap
    acc_overlap = None
    if 'accession' in old_f.columns and 'accession' in new_f.columns:
        acc_overlap = len(set(old_f['accession']).intersection(set(new_f['accession'])))

    # Feature set diffs
    old_hits = [c for c in old_f.columns if 'hit_count' in c]
    new_hits = [c for c in new_f.columns if 'hit_count' in c]
    only_old, only_new, same_order = compare_feature_sets(old_hits, new_hits)

    print(f"\n=== {label} {species} ===")
    print(f"Old rows: {len(old_f)} | New rows: {len(new_f)} | Accession overlap: {acc_overlap}")
    print(f"Hit features: old={len(old_hits)}, new={len(new_hits)}")
    print(f"Features only in OLD (count={len(only_old)}): {only_old[:10]}{' ...' if len(only_old) > 10 else ''}")
    print(f"Features only in NEW (count={len(only_new)}): {only_new[:10]}{' ...' if len(only_new) > 10 else ''}")
    print(f"Feature order identical: {same_order}")

    # Compare numeric differences on common features
    common_features = sorted(list(set(old_hits) & set(new_hits)))
    if common_features:
        diff_summary, n_comp = per_feature_numeric_diffs(old_f, new_f, common_features)
        print(f"Compared numeric diffs on {len(common_features)} common features over {n_comp} aligned samples.")
        print("Top differing features (up to 10):")
        print(diff_summary.head(10))
    else:
        print("No common features to compare numerically.")

    # Metadata sanity
    meta_old = summarize_metadata(old_df, is_train)
    meta_new = summarize_metadata(new_df, is_train)
    print("Old metadata summary:", meta_old)
    print("New metadata summary:", meta_new)

def main_compare():
    # TRAIN comparisons
    for species in species_mapping.keys():
        old_train = OLD_TRAIN_FULL_ROOT / species / f"{species}_full_rf_dataset.csv"
        new_train = find_first_existing(candidate_new_train_paths(species))
        old_df = load_df(old_train, "OLD TRAIN", species)
        new_df = load_df(new_train, "NEW TRAIN", species)
        analyze_pair(old_df, new_df, species, is_train=True)

    # TEST comparisons
    for species in species_mapping.keys():
        old_test = OLD_TEST_ROOT / species / f"{species}_full_rf_dataset.csv"
        new_test = candidate_new_test_paths(species)
        old_df = load_df(old_test, "OLD TEST", species)
        new_df = load_df(new_test, "NEW TEST", species)
        analyze_pair(old_df, new_df, species, is_train=False)

#if __name__ == "__main__":
#    main_compare()
"""
"""
new_df = pd.read_csv('/gpfs/scratch/jvaska/CAMDA_AMR/AMR_v2/dnabert/inference/outputs/rf_datasets/oof/run2/staphylococcus_aureus/staphylococcus_aureus_full_rf_dataset.csv')
old_df = pd.read_csv('/gpfs/scratch/jvaska/CAMDA_AMR/AMR_v2/dnabert/inference/outputs/rf_datasets/original_dataset/train/FULL/staphylococcus_aureus/staphylococcus_aureus_full_rf_dataset.csv')

new_accs = new_df['accession'].tolist()
old_accs = old_df['accession'].tolist()

print(len(new_accs))
print(len(old_accs))

print(len(set(new_accs)))
print(len(set(old_accs)))
"""
"""
# ---- Write NEW-only train accessions per species to ./missing_accs/{species}.txt ----
import os
from pathlib import Path
import pandas as pd

# Reuse existing constants if present; otherwise define them here
try:
    species_mapping
except NameError:
    species_mapping = {
        'klebsiella_pneumoniae': 0,
        'streptococcus_pneumoniae': 1,
        'escherichia_coli': 2,
        'campylobacter_jejuni': 3,
        'salmonella_enterica': 4,
        'neisseria_gonorrhoeae': 5,
        'staphylococcus_aureus': 6,
        'pseudomonas_aeruginosa': 7,
        'acinetobacter_baumannii': 8
    }

try:
    NEW_TRAIN_ROOT
except NameError:
    NEW_TRAIN_ROOT = Path("/gpfs/scratch/jvaska/CAMDA_AMR/AMR_v2/dnabert/inference/outputs/rf_datasets/oof/run2")
try:
    OLD_TRAIN_FULL_ROOT
except NameError:
    OLD_TRAIN_FULL_ROOT = Path("/gpfs/scratch/jvaska/CAMDA_AMR/AMR_v2/dnabert/inference/outputs/rf_datasets/original_dataset/train/FULL")

def _find_first_existing(paths):
    for p in paths:
        if p.exists():
            return p
    return None

def _candidate_new_train_paths(species):
    # Handle possible filename typo (_datase.csv vs _dataset.csv)
    return [
        NEW_TRAIN_ROOT / species / f"{species}_full_rf_dataset.csv",
        NEW_TRAIN_ROOT / species / f"{species}_full_rf_datase.csv",
    ]

def write_new_only_accessions():
    out_root = Path("./missing_accs")
    out_root.mkdir(parents=True, exist_ok=True)

    summary = []
    for species in species_mapping.keys():
        old_path = OLD_TRAIN_FULL_ROOT / species / f"{species}_full_rf_dataset.csv"
        new_path = _find_first_existing(_candidate_new_train_paths(species))

        if not old_path.exists():
            print(f"[WARN] Missing OLD train file for {species}: {old_path}")
            continue
        if new_path is None:
            print(f"[WARN] Missing NEW train file for {species} (checked both _dataset and _datase variants).")
            continue

        try:
            old_df = pd.read_csv(old_path, usecols=['accession'])
            new_df = pd.read_csv(new_path, usecols=['accession'])
        except Exception as e:
            print(f"[ERROR] Failed to read accessions for {species}: {e}")
            continue

        old_accs = set(map(str, old_df['accession'].dropna().astype(str)))
        new_accs = set(map(str, new_df['accession'].dropna().astype(str)))

        new_only = sorted(list(new_accs - old_accs))

        out_file = out_root / f"{species}.txt"
        with open(out_file, "w") as f:
            for acc in new_only:
                f.write(f"{acc}\n")

        print(f"[OK] {species}: wrote {len(new_only)} new-only accessions to {out_file}")
        summary.append((species, len(new_only)))

    # Optional: print a brief summary
    if summary:
        print("\nSummary (species -> new-only count):")
        for sp, n in summary:
            print(f"  {sp:25s}: {n}")
    else:
        print("No species processed or no new-only accessions found.")

# Uncomment to run directly when executing this script
#if __name__ == "__main__":
#    write_new_only_accessions()
"""
"""
import joblib
model = joblib.load('/gpfs/scratch/jvaska/CAMDA_AMR/AMR_v2/rf/models/rf/per_species/run2_hitsonly_skipaccs/neisseria_gonorrhoeae_rf_model.joblib')
feature_names = model.feature_names_in_
accs_to_skip = open('/gpfs/scratch/jvaska/CAMDA_AMR/AMR_v2/data_analysis/missing_accs/neisseria_gonorrhoeae.txt').readlines()
accs_to_skip = [acc.strip() for acc in accs_to_skip]
#are they in feature_names?
print([acc for acc in accs_to_skip if acc in feature_names])
print(len(accs_to_skip))
print(len(feature_names))
print(len(set(accs_to_skip).intersection(set(feature_names))))
print(len(set(accs_to_skip).difference(set(feature_names))))
"""
"""
species_mapping = {
        'klebsiella_pneumoniae': 0,
        'streptococcus_pneumoniae': 1,
        'escherichia_coli': 2,
        'campylobacter_jejuni': 3,
        'salmonella_enterica': 4,
        'neisseria_gonorrhoeae': 5,
        'staphylococcus_aureus': 6,
        'pseudomonas_aeruginosa': 7,
        'acinetobacter_baumannii': 8
    }

import pandas as pd
from pathlib import Path
from sklearn.feature_selection import mutual_info_classif
import numpy as np

# Reuse your existing species_mapping and paths:
# species_mapping, NEW_TRAIN_ROOT, OLD_TRAIN_FULL_ROOT already defined above

def _find_first_existing(paths):
    for p in paths:
        if p.exists():
            return p
    return None

def _candidate_new_train_paths(species):
    return [
        NEW_TRAIN_ROOT / species / f"{species}_full_rf_dataset.csv",
        NEW_TRAIN_ROOT / species / f"{species}_full_rf_datase.csv",
    ]

def load_train_pair(species):
    old_path = OLD_TRAIN_FULL_ROOT / species / f"{species}_full_rf_dataset.csv"
    new_path = _find_first_existing(_candidate_new_train_paths(species))
    if not old_path.exists() or new_path is None:
        print(f"[SKIP] Missing train files for {species}")
        return None, None
    try:
        old_df = pd.read_csv(old_path)
        new_df = pd.read_csv(new_path)
        return old_df, new_df
    except Exception as e:
        print(f"[ERROR] Reading train for {species}: {e}")
        return None, None

def label_mismatch_report(old_df, new_df, species):
    # Intersect by accession
    common = sorted(list(set(old_df['accession']) & set(new_df['accession'])))
    A = old_df.set_index('accession').loc[common]
    B = new_df.set_index('accession').loc[common]
    if 'ground_truth_phenotype' not in A.columns or 'ground_truth_phenotype' not in B.columns:
        print(f"[WARN] Missing ground_truth_phenotype in {species}")
        return

    la = A['ground_truth_phenotype'].astype(int)
    lb = B['ground_truth_phenotype'].astype(int)
    mismask = la != lb
    mismatches = la[mismask].index.tolist()

    print(f"\n=== LABEL MISMATCHES (TRAIN) {species} ===")
    print(f"Common accessions: {len(common)}")
    print(f"Label mismatches: {mismask.sum()}")
    if mismask.sum() > 0:
        print(f"Sample mismatched accessions (up to 20): {mismatches[:20]}")

    # Class proportions on common subset
    def proportions(s):
        vc = s.value_counts().to_dict()
        total = len(s)
        return {k: (v, v/total) for k,v in vc.items()}

    print("OLD label proportions on common:", proportions(la))
    print("NEW label proportions on common:", proportions(lb))

def duplicate_and_conflict_report(df, species, tag):
    # Duplicate accession rows
    dup_counts = df['accession'].value_counts()
    dups = dup_counts[dup_counts > 1].index.tolist()
    print(f"\n=== DUPLICATES ({tag}) {species} ===")
    print(f"Total duplicates: {len(dups)}")
    if len(dups) > 0:
        # Check conflicting labels among duplicates
        conflicts = []
        for acc in dups:
            labels = df.loc[df['accession'] == acc, 'ground_truth_phenotype'].dropna().unique().tolist()
            if len(labels) > 1:
                conflicts.append((acc, labels))
        print(f"Conflicting-label duplicates: {len(conflicts)}")
        if conflicts:
            print("Examples (up to 10):", conflicts[:10])

def mi_drift_report(old_df, new_df, species, top_k=100, compute=True):
    if not compute:
        return
    # Restrict to common accessions and common hit features
    common = sorted(list(set(old_df['accession']) & set(new_df['accession'])))
    A = old_df.set_index('accession').loc[common]
    B = new_df.set_index('accession').loc[common]

    hit_cols = [c for c in A.columns if 'hit_count' in c]
    hit_cols = [c for c in hit_cols if c in B.columns]
    if len(hit_cols) == 0:
        print(f"[WARN] No common hit features for {species}")
        return

    Xa = A[hit_cols].values
    Xb = B[hit_cols].values
    ya = A['ground_truth_phenotype'].astype(int).values
    yb = B['ground_truth_phenotype'].astype(int).values

    # Compute MI against their respective labels
    mia = mutual_info_classif(Xa, ya, discrete_features='auto', random_state=0)
    mib = mutual_info_classif(Xb, yb, discrete_features='auto', random_state=0)

    df_mi_a = pd.DataFrame({'feature': hit_cols, 'mi': mia}).sort_values('mi', ascending=False)
    df_mi_b = pd.DataFrame({'feature': hit_cols, 'mi': mib}).sort_values('mi', ascending=False)

    top_a = df_mi_a.head(top_k)['feature'].tolist()
    top_b = df_mi_b.head(top_k)['feature'].tolist()
    overlap = len(set(top_a) & set(top_b))

    print(f"\n=== MI DRIFT (TRAIN) {species} ===")
    print(f"Top-{top_k} overlap (OLD vs NEW): {overlap}/{top_k}")
    if overlap < top_k:
        # show top-10 MI diff by rank presence
        only_a = [f for f in top_a if f not in top_b][:10]
        only_b = [f for f in top_b if f not in top_a][:10]
        print(f"Top features only in OLD (up to 10): {only_a}")
        print(f"Top features only in NEW (up to 10): {only_b}")

def run_train_set_differences(top_k_mi=100, COMPUTE_MI=False):
    for species in species_mapping.keys():
        old_df, new_df = load_train_pair(species)
        if old_df is None or new_df is None:
            continue

        # Label mismatches on common accessions
        label_mismatch_report(old_df, new_df, species)

        # Duplicate + conflicting labels
        duplicate_and_conflict_report(old_df, species, tag="OLD")
        duplicate_and_conflict_report(new_df, species, tag="NEW")

        # MI drift on common accessions (optional, expensive)
        mi_drift_report(old_df, new_df, species, top_k=top_k_mi, compute=COMPUTE_MI)

# Uncomment to run:
#if __name__ == "__main__":
#    run_train_set_differences(top_k_mi=100, COMPUTE_MI=False)


import os
from pathlib import Path
import pandas as pd

# Reuse existing globals if present:
# species_mapping, NEW_TRAIN_ROOT, OLD_TRAIN_FULL_ROOT
def _find_first_existing(paths):
    for p in paths:
        if p.exists():
            return p
    return None

def _candidate_new_train_paths(species):
    # Tries both _dataset and the common typo _datase
    return [
        NEW_TRAIN_ROOT / species / f"{species}_full_rf_dataset.csv",
        NEW_TRAIN_ROOT / species / f"{species}_full_rf_datase.csv",
    ]

def write_label_mismatches():
    out_root = Path("./mismatches")
    out_root.mkdir(parents=True, exist_ok=True)

    summary = []
    for species in species_mapping.keys():
        old_path = OLD_TRAIN_FULL_ROOT / species / f"{species}_full_rf_dataset.csv"
        new_path = _find_first_existing(_candidate_new_train_paths(species))
        if not old_path.exists() or new_path is None:
            print(f"[SKIP] Missing train files for {species}")
            continue

        try:
            old_df = pd.read_csv(old_path, usecols=['accession', 'ground_truth_phenotype'])
            new_df = pd.read_csv(new_path, usecols=['accession', 'ground_truth_phenotype'])
        except Exception as e:
            print(f"[ERROR] Reading train for {species}: {e}")
            continue

        # Intersect and compare labels
        common = sorted(list(set(old_df['accession']) & set(new_df['accession'])))
        if len(common) == 0:
            print(f"[INFO] No common accessions for {species}")
            continue

        A = old_df.set_index('accession').loc[common]['ground_truth_phenotype'].astype(int)
        B = new_df.set_index('accession').loc[common]['ground_truth_phenotype'].astype(int)
        mismask = A != B
        mismatches = A[mismask].index.tolist()

        out_file = out_root / f"{species}.txt"
        with open(out_file, "w") as f:
            for acc in mismatches:
                f.write(f"{acc}\n")

        print(f"[OK] {species}: wrote {len(mismatches)} label-mismatched accessions to {out_file}")
        summary.append((species, len(mismatches)))

    if summary:
        print("\nSummary (species -> mismatch count):")
        for sp, n in summary:
            print(f"  {sp:25s}: {n}")
    else:
        print("No mismatches written (no species processed or no mismatches found).")

# Uncomment to run directly:
if __name__ == "__main__":
    write_label_mismatches()
"""

"""#check phenotype balance

train_df = pd.concat([pd.read_csv('/gpfs/scratch/jvaska/CAMDA_AMR/AMR_v2/dnabert/finetune/data/oof/run5/fold_0/dnabert_data/train.csv'), pd.read_csv('/gpfs/scratch/jvaska/CAMDA_AMR/AMR_v2/dnabert/finetune/data/oof/run5/fold_0/dnabert_data/dev.csv')])
print(train_df.head())
print(train_df['phenotype'].value_counts())

#ensure theres no accessions that are not in ./top_15p_features/{species}_top_15%_features.txt
for species in ['klebsiella_pneumoniae', 'streptococcus_pneumoniae', 'escherichia_coli', 'campylobacter_jejuni', 'salmonella_enterica', 'neisseria_gonorrhoeae', 'staphylococcus_aureus', 'pseudomonas_aeruginosa', 'acinetobacter_baumannii']:
    top_15p_features = open(f'./top_15p_features/{species}_top_15%_features.txt').readlines()
    top_15p_features = [line.strip() for line in top_15p_features]
    species_df = train_df[train_df['species'] == species]
    species_df['accession'].isin(top_15p_features).all()
    if not species_df['accession'].isin(top_15p_features).all():
        print(f"[ERROR] {species} has accessions not in top 15% features")
    else:
        print(f"[OK] {species} has all accessions in top 15% features")

#check query_id / phenotype balance
query_id_to_phenotype = train_df.groupby('query_id')['phenotype'].value_counts().to_dict()
print(query_id_to_phenotype)

#how many nan values in train_df
print(train_df.isnull().sum())

#number of duplicate sequences
print(train_df['sequence'].duplicated().sum())

"""







    
"""
eval_results = pd.read_csv('/gpfs/scratch/jvaska/CAMDA_AMR/AMR_v2/rf/eval_results/rf_models.csv')
print(eval_results.drop(['top_n_mi_score', 'model_type', 'recall', 'max_depth', 'min_samples_leaf'], axis=1).to_dict())"""
"""
#show lengths of all hit count features in each species' dataset
for species in species_mapping.keys():
    df = pd.read_csv(f'/gpfs/scratch/jvaska/CAMDA_AMR/AMR_v2/dnabert/inference/outputs/rf_datasets/run5_1_trainset/{species}/{species}_full_rf_dataset.csv')
    hit_cols = [c for c in df.columns if 'hit_count' in c]
    print(f'{species}: {len(hit_cols)} hit count features')
    for col in hit_cols:
        print(f'  {col}: {df[col].sum()} hits')"""
    


"""df = pd.read_csv('/gpfs/scratch/jvaska/CAMDA_AMR/AMR_v2/data_pipeline/datasets/run6_1500bp/sequence_based/per_antibiotic/train/full_sequence_dataset.csv')
print(df.head())
print(len(df))
df['len'] = df['sequence'].str.len()
print(df['len'].value_counts())"""

"""
#dbgwas info
df = pd.read_csv('/gpfs/scratch/jvaska/CAMDA_AMR/CAMDA_AMR/dbgwas/runs/2class_dbgwas_mixed/acinetobacter_baumannii/dbgwas_output/textualOutput/all_comps_nodes_info.tsv', sep='\t')
print(df.head())
"""

"""
df = pd.read_csv('/gpfs/scratch/jvaska/CAMDA_AMR/AMR_v2/dnabert/inference/outputs/rf_datasets/run6_alllogits/run6_1000bp_alllogits_fold_0/full/train/acinetobacter_baumannii/acinetobacter_baumannii_full_rf_dataset.csv')
print(f'NaNs in dataset:')
cols = [c for c in df.columns.tolist() if 'std' in c]
print(df[cols].isna().sum())
print()
cols = [c for c in df.columns.tolist() if 'mean' in c]
print(df[cols].isna().sum())
print()
cols = [c for c in df.columns.tolist() if 'sum' in c]
print(df[cols].isna().sum())
print()"""
"""
#for species in species_mapping.keys():
print('Train set duplicates:')
df = pd.read_csv(f'/gpfs/scratch/jvaska/CAMDA_AMR/AMR_v2/data_pipeline/datasets/sequence_based/per_antibiotic/train/full_sequence_dataset.csv')
print(f'Percent dupes: {(len(df) - len(df.drop_duplicates(subset='sequence'))) / len(df)}')



print()
print('Test set dupes:')
df = pd.read_csv('/gpfs/scratch/jvaska/CAMDA_AMR/AMR_v2/data_pipeline/datasets/sequence_based/per_antibiotic/test/full_sequence_dataset.csv')
print(f'Percent dupes: {(len(df) - len(df.drop_duplicates(subset='sequence'))) / len(df)}')

"""

"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from tqdm import tqdm

# Directory containing TSV files
dir_path = '/gpfs/scratch/jvaska/CAMDA_AMR/AMR_v2/data_pipeline/work/blast/hits/test/per_species'

# List to store all third column values
percids = []
query_ids = []

# Get all TSV files in the directory
tsv_files = glob.glob(os.path.join(dir_path, '*.tsv'))

print(f"Found {len(tsv_files)} TSV files")

# Process each TSV file
for tsv_file in tqdm(tsv_files):
    try:
        with open(tsv_file, 'r') as f:
            for line in f:
                # Skip empty lines
                if not line.strip():
                    continue
                
                # Split by tab and get the third column (index 2)
                columns = line.strip().split('\t')
                if len(columns) >= 3:
                    try:
                        # Convert to float and append
                        percid = float(columns[2])
                        percids.append(percid)
                        queryid = columns[0]
                        query_ids.append(queryid)
                    except ValueError:
                        # Skip if can't convert to float (e.g., header row)
                        continue
    except Exception as e:
        print(f"Error processing {tsv_file}: {e}")

# Convert to numpy array for easier calculations
percids = np.array(percids)
query_ids = np.array(query_ids)

#make df with query_id and percid
percids_df = pd.DataFrame({'query_id': query_ids, 'avg_perc_id': percids})
#get average percid for each query_id
percids_df = percids_df.groupby('query_id').mean().reset_index()
percids_df.to_csv('percids_df.csv', index=False)



# Calculate statistics
mean_val = np.mean(all_values)
mode_result = stats.mode(all_values, keepdims=True)
mode_val = mode_result.mode[0]
std_val = np.std(all_values)
num_100 = np.sum(all_values == 100)
percent_100 = (num_100 / len(all_values)) * 100


print(f"\nStatistics:")
print(f"Mean: {mean_val:.4f}")
print(f"Mode: {mode_val:.4f}")
print(f"Standard Deviation: {std_val:.4f}")
print(f"Min: {np.min(all_values):.4f}")
print(f"Max: {np.max(all_values):.4f}")
print(f'percent100 = {percent_100}')


# Create histogram with bin size of 0.5
min_val = np.min(all_values)
max_val = np.max(all_values)
bins = np.arange(min_val, max_val + 0.5, 0.5)


plt.figure(figsize=(12, 6))
plt.hist(all_values, bins=bins, edgecolor='black', alpha=0.7)
plt.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.4f}')
plt.axvline(mode_val, color='green', linestyle='--', linewidth=2, label=f'Mode: {mode_val:.4f}')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Distribution of Third Column Values from All TSV Files')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('tsv_column3_histogram.png', dpi=300, bbox_inches='tight')

# Save all values to a file
#np.savetxt('all_values.txt', all_values, fmt='%.4f')

print("\nHistogram saved as 'tsv_column3_histogram.png'")
plt.show()
"""

"""
for species in species_mapping.keys():
    print(species)
    df = pd.read_csv(f'/gpfs/scratch/jvaska/CAMDA_AMR/AMR_v2/dnabert/inference/outputs/rf_datasets/run5_1_trainset/run5_fold_0/full/train/{species}/{species}_full_rf_dataset.csv')
    hit_cols = [c for c in df.columns if 'hit_count' in c]
    print(df[hit_cols].shape)
"""
"""

df = pd.read_csv('/gpfs/scratch/jvaska/CAMDA_AMR/AMR_v2/dnabert/finetune/data/oof/run6_1000bp/fold_1/dnabert_data/train.csv')
print(len(df))
print(df['phenotype'].value_counts(normalize=True))
"""

"""
df = pd.read_csv('/gpfs/scratch/jvaska/CAMDA_AMR/AMR_v2/dnabert/finetune/data/oof/run7/fold_0/dnabert_data/train.csv')
print(f'Percent dupes: {(len(df) - len(df.drop_duplicates(subset='sequence'))) / len(df)}')
print(len(df))
print(df['phenotype'].value_counts())
"""
"""
data = {}
for species in species_mapping.keys():
    df = pd.read_csv(f'/gpfs/scratch/jvaska/CAMDA_AMR/CAMDA_AMR/dbgwas/runs/2class_dbgwas_filtered/{species}/dbgwas_output/textualOutput/all_comps_nodes_info.tsv', sep='\t', index_col=False)
    #print(df.head(2))
    #print(df['Significant?'].value_counts())
    print(species)
    print(df[df['p-value'] < 0.05]['EstEffect'].map(lambda x: -1 if x < 0 else 1).value_counts())
    data[species] = {'Res': int(df[df['p-value'] < 0.05]['EstEffect'].map(lambda x: -1 if x < 0 else 1).value_counts()[-1]), 'Sus': int(df[df['p-value'] < 0.05]['EstEffect'].map(lambda x: -1 if x < 0 else 1).value_counts()[1] if 1 in df[df['p-value'] < 0.05]['EstEffect'].map(lambda x: -1 if x < 0 else 1).value_counts().index else 0)}

print(data)
"""
"""
df = pd.read_csv('/gpfs/scratch/jvaska/CAMDA_AMR/AMR_v2/data_pipeline/datasets/run8_70pct/sequence_based/per_antibiotic/train/full_sequence_dataset.csv')
print(len(df))
print(len(df.drop_duplicates(subset='sequence')))
print(df.drop_duplicates(subset='sequence')['phenotype'].value_counts())
#print(df['phenotype'].value_counts())
#print(df.drop_duplicates(subset='sequence')['phenotype'].value_counts())
"""
"""
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score

f1_threshold = .75

def calculate_f1(group):
    f1_sus = f1_score(group['phenotype'], group['pred'], pos_label=1, zero_division=0)
    f1_res = f1_score(group['phenotype'], group['pred'], pos_label=0, zero_division=0)
    macro_f1 = (f1_sus + f1_res) / 2
    return macro_f1

#get accuracy for each gene (query_id) from max of pred_res and pred_sus logits columns and grount truth 'phenotype' column, where 'phenotype' has res=0 and sus=1
seq_preds_df = pd.concat([pd.read_csv('/gpfs/scratch/jvaska/CAMDA_AMR/AMR_v2/dnabert/inference/outputs/preds/run6_1000bp_alllogits_fold_0/full/train/full_preds.csv'), pd.read_csv('/gpfs/scratch/jvaska/CAMDA_AMR/AMR_v2/dnabert/inference/outputs/preds/run6_1000bp_alllogits_fold_1/full/train/full_preds.csv'), pd.read_csv('/gpfs/scratch/jvaska/CAMDA_AMR/AMR_v2/dnabert/inference/outputs/preds/run6_1000bp_alllogits_fold_2/full/train/full_preds.csv')])

seq_preds_df['pred'] = (seq_preds_df['pred_res'] < seq_preds_df['pred_sus']).astype(int) #1 is sus, 0 is res
f1_by_query = seq_preds_df.groupby('query_id').apply(calculate_f1).reset_index(name='f1_score')
f1_by_query = f1_by_query.sort_values('f1_score', ascending=False)

count_by_query = seq_preds_df['query_id'].value_counts().reset_index()
count_by_query.columns = ['query_id', 'sample_count']

result = pd.merge(f1_by_query, count_by_query, on='query_id')

print("Top 10 query_ids by prediction f1:")
print(result.head(20))

high_acc_df = result[result['f1_score'] >= f1_threshold]
print(f"Number of query_ids with >= {f1_threshold} f1 score:", len(high_acc_df))
print('Number query_ids with >0.9 f1 score:', len(result[result['f1_score'] > 0.9]))

print(f'Sample count stats for query_ids with >= {f1_threshold} f1 score:')
print(high_acc_df['sample_count'].describe())
print()
print(f'Average phenotype and prediction for query_ids with >= {f1_threshold} f1 score:')
print(seq_preds_df[seq_preds_df['query_id'].isin(high_acc_df['query_id'])].groupby('query_id').agg({'phenotype': 'mean', 'pred': 'mean'}))

print()

#save all query_ids with >= f1_threshold f1 score to a plain text file
with open('top75_f1_queryids.txt', 'w') as f:
    for query_id in high_acc_df['query_id']:
        f.write(f'{query_id}\n')

"""
"""

#get RF feature importances:

dfs = []
import joblib
import numpy as np
models_dir = '/gpfs/scratch/jvaska/CAMDA_AMR/AMR_v2/rf/models/rf/per_species/run11_NOFILTER_ALL'
for model_file in os.listdir(models_dir):
    if model_file.endswith('.joblib'):
        model_path = os.path.join(models_dir, model_file)
        model = joblib.load(model_path)
        feature_importances = model.feature_importances_
        feature_names = model.feature_names_in_
        print(feature_names)
        query_ids = list(set([feature_name.split('_', 2)[0] + '_' + feature_name.split('_', 2)[1] for feature_name in feature_names]))
        hit_count_importances = []
        pred_resistant_importances = []
        for query_id in query_ids:
            #for each query_id, get corresponding hit count and pred resistant importances
            if f'{query_id}_hit_count' in feature_names:
                #print(f'{query_id}_hit_count')
                hit_count_importances.append(feature_importances[np.where(feature_names == f'{query_id}_hit_count')[0][0]])
            else:
                hit_count_importances.append(None)
            if f'{query_id}_res_int_pred_resistant' in feature_names:
                #print(f'{query_id}_res_int_pred_resistant')
                pred_resistant_importances.append(feature_importances[np.where(feature_names == f'{query_id}_res_int_pred_resistant')[0][0]])
            else:
                pred_resistant_importances.append(None)
            
        df = pd.DataFrame({'query_id': query_ids,
                           'hit_count_rf_importance': hit_count_importances,
                           'pred_resistant_rf_importance': pred_resistant_importances})
        #normalize importances to max hit count importance per-species
        df['hit_count_rf_importance'] = df['hit_count_rf_importance'] / df['hit_count_rf_importance'].max()
        df['pred_resistant_rf_importance'] = df['pred_resistant_rf_importance'] / df['hit_count_rf_importance'].max()
        
        dfs.append(df)


importances_df = pd.concat(dfs, axis=0)
print(importances_df.head(20))

print(importances_df.sort_values('hit_count_rf_importance', ascending=False).head(20))


#get ratio of duplicate seqs

# *****THIS IS HOW WE GOT ROC VALUES FOR CANDIDATE GENES
#calculate ROC AUC for each query_id for dnabert preds and for hit counts. Get list of query_ids that aucroc(dnabert) > aucroc(hit counts)
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score

def safe_auc(y_true, y_score): #we need an agnostic auc, because sometimes more hits means less resistant, and sometimes more hits means more resistant
    # Skip if constant feature or single-class labels
    if len(np.unique(y_score)) == 1 or len(np.unique(y_true)) == 1:
        return 0.5
    auc = roc_auc_score(y_true, y_score)
    return max(auc, 1 - auc)  # make direction-agnostic

def calculate_f1(group):
    f1_sus = f1_score(group['phenotype'], group['pred'], pos_label=1, zero_division=0)
    f1_res = f1_score(group['phenotype'], group['pred'], pos_label=0, zero_division=0)
    macro_f1 = (f1_sus + f1_res) / 2
    return macro_f1

seq_preds_df = pd.concat([pd.read_csv('/gpfs/scratch/jvaska/CAMDA_AMR/AMR_v2/dnabert/inference/outputs/preds/run6_1000bp_alllogits_fold_0/full/train/full_preds.csv'), pd.read_csv('/gpfs/scratch/jvaska/CAMDA_AMR/AMR_v2/dnabert/inference/outputs/preds/run6_1000bp_alllogits_fold_1/full/train/full_preds.csv'), pd.read_csv('/gpfs/scratch/jvaska/CAMDA_AMR/AMR_v2/dnabert/inference/outputs/preds/run6_1000bp_alllogits_fold_2/full/train/full_preds.csv')])
seq_preds_df['pred'] = (seq_preds_df['pred_res'] < seq_preds_df['pred_sus']).astype(int) #1 is sus, 0 is res *if using integer preds (otherwise skip this and use pred_res or pred_sus)

roc_auc_dnabert = seq_preds_df.groupby('query_id').apply(lambda x: roc_auc_score(x['phenotype'], x['pred_sus'])).reset_index(name='roc_auc_dnabert')
ratio_res = seq_preds_df.groupby('query_id').apply(lambda x: (x['phenotype'].value_counts()[0] / len(x)) if len(x['phenotype'].value_counts()) == 2 else 0).reset_index(name='ratio_res_sus') #ratio of resistant to susceptible, closer to 1 = even, <1 = more sus,  >1 = more res
num_res = seq_preds_df.groupby('query_id').apply(lambda x: x['phenotype'].value_counts()[0] if len(x['phenotype'].value_counts()) == 2 else 0).reset_index(name='num_res')
num_sus = seq_preds_df.groupby('query_id').apply(lambda x: x['phenotype'].value_counts()[1] if len(x['phenotype'].value_counts()) == 2 else 0).reset_index(name='num_sus')
counts = seq_preds_df.groupby('query_id').apply(lambda x: len(x)).reset_index(name='counts')
accuracies = seq_preds_df.groupby('query_id').apply(lambda x: (x['pred'] == x['phenotype']).mean()).reset_index(name='accuracy')
macro_f1s = seq_preds_df.groupby('query_id').apply(lambda x: calculate_f1(x)).reset_index(name='macro_f1')
res_f1s = seq_preds_df.groupby('query_id').apply(lambda x: f1_score(x['phenotype'], x['pred'], pos_label=0, zero_division=0)).reset_index(name='res_f1')
sus_f1s = seq_preds_df.groupby('query_id').apply(lambda x: f1_score(x['phenotype'], x['pred'], pos_label=1, zero_division=0)).reset_index(name='sus_f1')
percids_df = pd.read_csv('percids_df.csv')


#get ratio of duplicate seqs within each group

dup_ratio = seq_preds_df.groupby('query_id').apply(lambda x: (x['sequence'].duplicated().sum() / len(x))).reset_index(name='dup_ratio')

merged_df = pd.merge(roc_auc_dnabert, ratio_res, on='query_id', how='left')
#merged_df = pd.merge(merged_df, macro_f1s, on='query_id', how='left')
merged_df = pd.merge(merged_df, res_f1s, on='query_id', how='left')
merged_df = pd.merge(merged_df, sus_f1s, on='query_id', how='left')
merged_df = pd.merge(merged_df, accuracies, on='query_id', how='left')
merged_df = pd.merge(merged_df, counts, on='query_id', how='left')
merged_df = pd.merge(merged_df, num_res, on='query_id', how='left')
merged_df = pd.merge(merged_df, num_sus, on='query_id', how='left')
merged_df = pd.merge(merged_df, dup_ratio, on='query_id', how='left')
merged_df = pd.merge(merged_df, percids_df, on='query_id', how='left')
merged_df = pd.merge(merged_df, importances_df, on='query_id', how='left')

merged_df.to_csv('gene_analysis.csv', index=False)

print(merged_df.sort_values('roc_auc_dnabert', ascending=False).head(20))
print(merged_df['roc_auc_dnabert'].describe())

print()

print(merged_df.sort_values('pred_resistant_rf_importance', ascending=False).head(20))
roc_thresh = 0.7
#get list of query_ids with aucroc(dnabert) > roc_thresh
high_auc_query_ids = roc_auc_dnabert[roc_auc_dnabert['roc_auc_dnabert'] > roc_thresh]['query_id'].tolist()
print(f'Number of query_ids with aucroc(dnabert) > {roc_thresh}: {len(high_auc_query_ids)}')
#print(high_auc_query_ids)
#to txt file
#with open('high_auc_query_ids.txt', 'w') as f:
#    for query_id in high_auc_query_ids:
#        f.write(f'{query_id}\n')

#roc_auc_hit_count = seq_preds_df.groupby('query_id').apply(lambda x: safe_auc(x['phenotype'], x['hit_count'])).reset_index(name='roc_auc_hit_count')

#merge
#merged_df = pd.merge(roc_auc_dnabert, roc_auc_hit_count, on='query_id')

#merged_df['auc_diff'] = merged_df['roc_auc_dnabert'] - merged_df['roc_auc_hit_count']

#print(merged_df.sort_values('auc_diff', ascending=False).head(20))

#how many auc_diff positive and how many negative
#merged_df['direction'] = merged_df['auc_diff'].apply(lambda x: 'positive' if x > 0 else 'negative')
#print(merged_df['direction'].value_counts())

"""

"""
# **** NOW LETS ANALYZE RESULTS AND GET THE GENES THAT INCREASE RF ACCURACY THE MOST

acc_threshold = .015
numpredres_df = pd.read_csv("/gpfs/scratch/jvaska/CAMDA_AMR/AMR_v2/rf/scripts/results/numpredres_results.csv")
alllogits_df = pd.read_csv("/gpfs/scratch/jvaska/CAMDA_AMR/AMR_v2/rf/scripts/results/alllogits_results.csv")

numpredres_df['dtype'] = 'numpredres'
alllogits_df['dtype'] = 'alllogits'

combined_df = pd.concat([numpredres_df, alllogits_df])
combined_df = combined_df[combined_df['accuracy_difference'] > acc_threshold] #filter for genes that increase accuracy by acc_threshold or more
#get top gene from each species
#combined_df = combined_df.sort_values(by="accuracy_difference", ascending=False).drop_duplicates(subset='species', keep='first')

combined_df = combined_df.sort_values(by="accuracy_difference", ascending=False).drop_duplicates(subset='query_id', keep='first')

print(combined_df.head(20))
print(combined_df['accuracy_difference'].describe())
print(combined_df['species'].value_counts())
print(len(combined_df))

combined_df.to_csv('high_acc_genes.csv', index=False)

"""

#get preds
preds_path = '/gpfs/scratch/jvaska/CAMDA_AMR/AMR_v2/rf/eval_results/hitsonly_80pct'
preds = pd.concat([pd.read_csv(f'{preds_path}/fold_1_preds.csv'), pd.read_csv(f'{preds_path}/fold_2_preds.csv'), pd.read_csv(f'{preds_path}/fold_0_preds.csv')])
false_positives = preds[(preds['true'] == 1) & (preds['pred'] == 0)]
accs = false_positives['accession']

print(f'num false positive accs: {len(accs)}')

#concat all matrixes
base_path = '/gpfs/scratch/jvaska/CAMDA_AMR/AMR_v2/dnabert/inference/outputs/rf_datasets/run6_1000bp_numpredres'
dfs = []
for fold in range(3):
    for species in species_mapping.keys():
        df = pd.read_csv(f'{base_path}/run6_1000bp_fold_{fold}/full/train/{species}/{species}_full_rf_dataset.csv')
        dfs.append(df)
full_df = pd.concat(dfs)
hit_cols = [col for col in full_df.columns if 'hit_count' in col]
hit_cols.extend(['accession', 'species', 'ground_truth_phenotype'])
full_df = full_df[hit_cols]
full_df = full_df[full_df['accession'].isin(accs)]
print(f'num false positive accs in full df: {len(full_df)}')

#count up query_ids (column) counts
#sum across column axis
col_sums = full_df.drop(columns=['accession', 'species', 'ground_truth_phenotype']).sum(axis=0)
print(col_sums)

#save col_sums to csv
col_sums.to_csv('qids_by_cm/false_positive_counts.csv')

#full_df.to_csv('full_sequence_dataset.csv', index=False)
