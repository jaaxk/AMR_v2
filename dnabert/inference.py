"""
Run inference on a specified DNABERT model
This script is mean to be used with datasets formatted by the data_pipeline/pipeline.py script with the --model_type sequence_based option
Output is a dataset for input to a random forest model, with columns: accession, species, antibiotic, hit count for each feature, predicted resistant for each feature, ground truth phenotype (if train) TODO: add option for consensus prediction

Key Options:
    - "output_format": "random_forest" or "consensus"
    - "grouping": "full", "per_antibiotic", or "per_species"

Paths to specigy:
    - "dataset_dir": path to directory containing datasets formatted by data_pipeline/pipeline.py
    - "model_path": path to DNABERT model
"""

# global parameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_MAX_LENGTH = 250 # should be 1/4 * sequence length
BATCH_SIZE = 32

# global lists/mappings
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

species_to_antibiotic = {
    'klebsiella_pneumoniae': 'GEN',
    'streptococcus_pneumoniae': 'ERY',
    'escherichia_coli': 'GEN',
    'campylobacter_jejuni': 'TET',
    'salmonella_enterica': 'GEN',
    'neisseria_gonorrhoeae': 'TET',
    'staphylococcus_aureus': 'ERY',
    'pseudomonas_aeruginosa': 'CAZ',
    'acinetobacter_baumannii': 'CAZ',
}

antibiotic_mapping = {'GEN': 0, 
    'ERY': 1,
    'CAZ': 2,
    'TET': 3,
}


# methods:
def load_model(model_path):
    model = transformers.AutoModelForSequenceClassification.from_pretrained(
        model_path,
        num_labels=2,
        trust_remote_code=True,
    )
    model.eval()
    model.to(device)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_path,
        model_max_length=MODEL_MAX_LENGTH,
        padding_side="right",
        use_fast=True,
        trust_remote_code=True,
    )
    return model, tokenizer


def inference(dataset_path, preds_path, model, tokenizer):
    #write preds as new column 'pred_phenotype' and write to new csv at preds_path
    df = pd.read_csv(dataset_path)
    pred_phenos = []
    with torch.no_grad():
        for start in tqdm(range(0, len(df), BATCH_SIZE)):
            batch = df[start:start+BATCH_SIZE]
            enc = tokenizer(list(batch['sequence']), padding=True, truncation=True, return_tensors='pt', max_length=MODEL_MAX_LENGTH)
            #num_hits = batch['num_hits'].values #not using num_hits and species for DNABERT pass anymore, these should be given to the RF model
            #species = batch['species'].values
            inputs = {
                'input_ids': enc['input_ids'].to(device),
                'attention_mask': enc['attention_mask'].to(device),
                #'num_hits': torch.tensor(num_hits, dtype=torch.float32).unsqueeze(1).to(device),
                #'species': torch.tensor(species, dtype=torch.long).to(device).unsqueeze(1).to(device),
            }
            logits = model(**inputs).logits
            preds = torch.argmax(logits, dim=-1).cpu().numpy()
            pred_phenos.extend([int(x) for x in preds.flatten()])

    df['pred_phenotype'] = pred_phenos
    df.to_csv(preds_path, index=False)

    return pred_phenos


def get_rf_dataset(preds_path, output_path, train):
    #make dataset with columns: accession, species (label), antibiotic (label), feature 1 hit count, feature 1 pred_resistant, ..., feature n hit count, feature n pred_resistant, ground truth phenotype (if train)
    # will be len(query_id.unique) + 3 (4 if train) columns
    df = pd.read_csv(preds_path)
    features = df['query_id'].unique().tolist()
    hit_count_features = [f + '_hit_count' for f in features]
    pred_resistant_features = [f + '_pred_resistant' for f in features]
    
    
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        #write header
        if train:
            header = ['accession', 'species', 'antibiotic'] + hit_count_features + pred_resistant_features + ['ground_truth_phenotype']
        else:
            header = ['accession', 'species', 'antibiotic'] + hit_count_features + pred_resistant_features

        writer.writerow(header)

        #iterate through accessions
        for accession in df['accession'].unique():
            #get species and antibiotic
            accession_df = df[df['accession'] == accession]
            species = accession_df['species'].iloc[0]
            antibiotic = accession_df['antibiotic'].iloc[0]
            #iterate through query_ids (features)
            query_id_to_hitcount = {}
            query_id_to_predresistant = {}
            for query_id in accession_df['query_id'].unique().tolist():
                query_id_df = accession_df[accession_df['query_id'] == query_id] #this df now only has a single query id for a single accession
                #get number predicted resistant for each query id
                num_pred_resistant = query_id_df['pred_phenotype'].value_counts()[1] # need to flip because Susceptible is 1
                query_id_to_hitcount[f'{query_id}_hit_count'] = query_id_df['hit_count'].iloc[0]
                query_id_to_predresistant[f'{query_id}_pred_resistant'] = num_pred_resistant

            #fill feature lists to match length of header
            hit_count_features_temp = []
            pred_resistant_features_temp = []
            for feature in hit_count_features:
                if feature in query_id_to_hitcount.keys():
                    hit_count_features_temp.append(query_id_to_hitcount[feature])
                else:
                    hit_count_features_temp.append(0)
            for feature in pred_resistant_features:
                if feature in query_id_to_predresistant.keys():
                    pred_resistant_features_temp.append(query_id_to_predresistant[feature])
                else:
                    pred_resistant_features_temp.append(0)

            #write row
            row = [accession, species, antibiotic] + hit_count_features_temp + pred_resistant_features_temp
            if train:
                row.append(accession_df['phenotype'].iloc[0])
            writer.writerow(row)

def get_consensus_preds(preds_path):
    pass

            

                

                
                
            
        
        
    



# imports
import pandas as pd
import argparse
import torch
import transformers
from tqdm import tqdm



def main():

    #parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_format', type=str, choices=['random_forest', 'consensus'], default='random_forest')
    parser.add_argument('--dataset_dir', type=str, help='Path to directory containing datasets formatted by data_pipeline/pipeline.py', default=None)
    parser.add_argument('--model_path', type=str, help='Path to DNABERT model, should take in ONLY "sequence", no num_hits, species, or antibiotic', default=None)
    parser.add_argument('--grouping', type=str, help='Grouping', choices=['full', 'per_species', 'per_antibiotic'], default='per_species')
    parser.add_argument('--train', type=bool, default=False)
    args = parser.parse_args()
    

    #load model
    model, tokenizer = load_model(args.model_path) #this current setup loads a full model and runs inference on all species/antibiotics on the same model (taking in only sequence). TODO: enable per-species/per-antibiotic models

    #loop through dataset depending on grouping
    if args.grouping == 'per_species':
        for species in species_list:
            dataset_path = os.path.join(args.dataset_dir, species, f'{species}_sequence_dataset.csv') #load sequence dataset
            if args.train:
                preds_path = f'./outputs/preds/{args.grouping}/train/{species}_preds.csv'
            else:
                preds_path = f'./outputs/preds/{args.grouping}/test/{species}_preds.csv'
            if not os.path.exists(os.path.dirname(preds_path)):
                os.makedirs(os.path.dirname(preds_path))
            inference(dataset_path, preds_path, model, tokenizer)
            if args.output_format == 'random_forest':
                output_path = f'./outputs/rf_datasets/per_species/{species}_rf_dataset.csv'
                if not os.path.exists(os.path.dirname(output_path)):
                    os.makedirs(os.path.dirname(output_path))
                get_rf_dataset(preds_path, output_path, args.train)
            elif args.output_format == 'consensus':
                get_consensus_preds(preds_path)

    if args.grouping == 'per_antibiotic':
        for antibiotic in antibiotic_list:
            dataset_path = os.path.join(args.dataset_dir, antibiotic, f'{antibiotic}_sequence_dataset.csv') #load sequence dataset
            if args.train:
                preds_path = f'./outputs/preds/{args.grouping}/train/{antibiotic}_preds.csv'
            else:
                preds_path = f'./outputs/preds/{args.grouping}/test/{antibiotic}_preds.csv'
            if not os.path.exists(os.path.dirname(preds_path)):
                os.makedirs(os.path.dirname(preds_path))
            inference(dataset_path, preds_path, model, tokenizer)
            if args.output_format == 'random_forest':
                output_path = f'./outputs/rf_datasets/per_antibiotic/{antibiotic}_rf_dataset.csv'
                if not os.path.exists(os.path.dirname(output_path)):
                    os.makedirs(os.path.dirname(output_path))
                get_rf_dataset(preds_path, output_path, args.train)

            elif args.output_format == 'consensus':
                get_consensus_preds(preds_path)
            

    pass

if __name__ == "__main__":
    main()