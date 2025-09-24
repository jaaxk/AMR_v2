"""
Use pretrained RF models from train_rf.py to predict phenotypes and fill out official CAMDA testing template
For evaluation on official site

Paths to specify:
    --testing_template: path to testing template
    --models_dir: directory containing RF models
        - should be at ./models/rf/{grouping}/{model_name}
"""

# imports
import argparse
import pandas as pd
import joblib
import os
from tqdm import tqdm

# lists/mappings
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
phenotype_mapping = {1: 'Susceptible', 0: 'Resistant'}

def filter_feature_type(X, feature_type):
    """ Remove hit_count or pred_resistant columns based on feature_type,
    as well as other unnecessary columns like accession, species, antibiotic, ground_truth_phenotype,
    to be used for X dataset (features) """

    X = X.drop(['species', 'antibiotic'], axis=1) # we dont need species and antibiotic if were doing per-species models (all the same)
    extra_cols_to_keep = ['accession']


    if feature_type == 'dnabert':
        cols_to_keep = [col for col in X.columns if 'pred_resistant' in col]
        cols_to_keep.extend(extra_cols_to_keep)
        X = X[cols_to_keep]
    elif feature_type == 'hits':
        cols_to_keep = [col for col in X.columns if 'hit_count' in col]
        cols_to_keep.extend(extra_cols_to_keep)
        X = X[cols_to_keep]
    elif feature_type == 'both':
        return X
    else:
        raise ValueError(f"Invalid feature type: {feature_type}")

    return X

def filter_top_n_mi(X, species, models_dir):
    
    path_to_load = os.path.join(models_dir, 'mi_features', f'{species}_mi_features.txt')
    if os.path.exists(path_to_load):
        with open(path_to_load, 'r') as f:
            features_to_keep = [line.strip() for line in f]
            features_to_keep.append('accession')
        X = X[features_to_keep]
    else:
        print(f'No MI features found for {species}, using all features')
        
    return X

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, help='Name of DNABERT model used', default=None)
    parser.add_argument('--grouping', type=str, choices=['full', 'per_antibiotic', 'per_species'], help='Grouping for models (per_species=9 models, per_antibiotic=4 models, full=1 model)', default='per_species')
    parser.add_argument('--testing_template', type=str, help='Path to testing template', default='../data_pipeline/data/metadata/testing_template.csv')
    parser.add_argument('--models_dir', type=str, help='Directory containing RF models', default=None)
    parser.add_argument('--test_dataset_dir', type=str, help='Path to directory containing test datasets formatted by dnabert/inference/inference.py')
    parser.add_argument('--out_path', type=str, help='Path to save filled template', default=None)
    parser.add_argument('--feature_type', type=str, choices=['dnabert', 'hits', 'both'], help='Feature type for RF models', default='both')
    args = parser.parse_args()

    if args.models_dir is None:
        args.models_dir = os.path.join('models', 'rf', args.grouping, args.model_name)
    if args.out_path is None:
        args.out_path = os.path.join('predictions', f'predictions_{args.model_name}_{args.grouping}.csv')

    testing_template = pd.read_csv(args.testing_template)
    acc_to_prediction = {}
    if args.grouping == 'per_species':
        for species in tqdm(species_list, desc='Running per-species RF inference'):
            model_path = os.path.join(args.models_dir, f"{species}_rf_model.joblib")
            model = joblib.load(model_path)
            df = pd.read_csv(os.path.join(args.test_dataset_dir, species, f"{species}_full_rf_dataset.csv"))
            df = filter_feature_type(df, args.feature_type)
            df = filter_top_n_mi(df, species, args.models_dir)
            preds = model.predict(df.drop(['accession'], axis=1))
            assert len(preds) == len(df)
            for acc, pred in zip(df['accession'], preds):
                acc_to_prediction[acc] = pred

        testing_template['phenotype'] = testing_template['accession'].map(acc_to_prediction)
        testing_template['phenotype'] = testing_template['phenotype'].map(phenotype_mapping)
        testing_template['phenotype'] = testing_template['phenotype'].fillna('Resistant')
        testing_template['measurement_value'] = 0
        testing_template.to_csv(args.out_path, index=None)
        print(f"Saved predictions to {args.out_path}")

if __name__ == '__main__':
    main()
