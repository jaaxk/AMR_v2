import pandas as pd
import argparse
import joblib
import os


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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models_dir", type=str, default="/gpfs/scratch/jvaska/CAMDA_AMR/AMR_v2/rf/models/rf/per_species/train_new_test_new_flip") #this did >87% accuracy on independent validation, and from new set
    parser.add_argument("--out_dir", type=str, default='../top_15p_features')
    parser.add_argument("--top_n_percent", type=int, default=15)
    args = parser.parse_args()

    # get all feature importances
    for species in species_list:
        model_path = f"{args.models_dir}/{species}_rf_model.joblib"
        model = joblib.load(model_path)
        df = pd.DataFrame(model.feature_importances_, index=model.feature_names_in_, columns=['importance'])
        df = df.sort_values(by='importance', ascending=False)
        features_to_get = len(df) * args.top_n_percent // 100
        top_features = df.head(features_to_get).index.tolist()
        os.makedirs(args.out_dir, exist_ok=True)
        with open(f"{args.out_dir}/{species}_top_{args.top_n_percent}%_features.txt", 'w') as f:
            for feature in top_features:
                f.write(f"{feature}\n")
        

    

if __name__ == "__main__":
    main()