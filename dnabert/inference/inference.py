"""
Run inference on a specified DNABERT model
This script is mean to be used with datasets formatted by the data_pipeline/pipeline.py script (or from oof_1.py) with the --model_type sequence_based option
Output is a dataset for input to a random forest model, with columns: accession, species, antibiotic, hit count for each feature, predicted resistant for each feature, ground truth phenotype (if train)

Key Options:
    - "output_format": "random_forest"
    - "grouping": "full", "per_antibiotic", or "per_species" - this is the grouping that the DNABERT models were trained on, output will always be per-species datasets here

Paths to specigy:
    - "dataset_dir": path to directory containing datasets formatted by data_pipeline/pipeline.py - should be full, per_species, or per_antibiotic dataset(s) according to grouping
    - "model_path": path to DNABERT model
"""

# imports
import pandas as pd
import argparse
import torch
import transformers
from tqdm import tqdm
import os
import csv
from Bio import SeqIO
import json

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

antibiotic_to_species ={
    'GEN': ['klebsiella_pneumoniae', 'escherichia_coli', 'salmonella_enterica'],
    'ERY': ['streptococcus_pneumoniae', 'staphylococcus_aureus'],
    'CAZ': ['pseudomonas_aeruginosa', 'acinetobacter_baumannii'],
    'TET': ['neisseria_gonorrhoeae', 'campylobacter_jejuni'],
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
        


def inference(dataset_path, preds_path, model, tokenizer, return_logits):
    #write preds as new columns 'pred_res' and 'pred_sus' and write to new csv at preds_path
    df = pd.read_csv(dataset_path)
    print(f"NUMBER OF NULL SEQUENCE ROWS: {df['sequence'].isnull().sum()}, dropping these rows")
    df = df.dropna(subset=['sequence'])
    pred_res = []
    pred_sus = []
    with torch.no_grad():
        for start in tqdm(range(0, len(df), BATCH_SIZE), desc='Running DNABERT inference'):
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
            #if return_logits is not None:
            preds_res = torch.softmax(logits, dim=-1).cpu().numpy()[:, 0]  # take only resistant class
            preds_sus = torch.softmax(logits, dim=-1).cpu().numpy()[:, 1]  # take only susceptible class
            pred_res.extend([float(x) for x in preds_res.flatten()])
            pred_sus.extend([float(x) for x in preds_sus.flatten()])
            #else:   
            #preds = torch.argmax(logits, dim=-1).cpu().numpy()

            #pred_res.extend([int(x) for x in preds.flatten()])

    df['pred_res'] = pred_res
    #if return_logits is not None:
    df['pred_sus'] = pred_sus
    df.to_csv(preds_path, index=False)


def get_rf_dataset(preds_path, output_path, train, sig_seqs_path, accessions, return_logits):
    #make dataset with columns: accession, species (label), antibiotic (label), feature 1 hit count, feature 1 pred_resistant, ..., feature n hit count, feature n pred_resistant, ground truth phenotype (if train)
    # will be len(query_id.unique) + 3 (4 if train) columns
    features = [record.id for record in SeqIO.parse(sig_seqs_path, 'fasta')] #get list of all query_ids in sig seqs fasta file for consistency between train and test datasets
    #print(f'len features: {len(features)}')
    #print(f'first 5 features: {features[:5]}')
    df = pd.read_csv(preds_path)
    #filter preds to only include accessions in accessions list 
    df = df[df['accession'].isin(accessions)]
    hit_count_features = [f + '_hit_count' for f in features]
    if return_logits != 'all':
        pred_resistant_features = [f + '_pred_resistant' for f in features]
    else:
        pred_resistant_features = [f + '_sus_sum_pred_resistant' for f in features] + \
        [f + '_sus_mean_pred_resistant' for f in features] + [f + '_sus_std_pred_resistant' for f in features] + \
        [f + '_res_sum_pred_resistant' for f in features] + [f + '_res_mean_pred_resistant' for f in features] + \
        [f + '_res_std_pred_resistant' for f in features]
    
    
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        #write header
        if train:
            header = ['accession', 'species', 'antibiotic'] + hit_count_features + pred_resistant_features + ['ground_truth_phenotype']
        else:
            header = ['accession', 'species', 'antibiotic'] + hit_count_features + pred_resistant_features

        writer.writerow(header)

        #iterate through accessions
        for accession in tqdm(df['accession'].unique(), desc='Generating RF dataset'):
            #get species and antibiotic
            accession_df = df[df['accession'] == accession]
            species = accession_df['species'].iloc[0]
            antibiotic = accession_df['antibiotic'].iloc[0]
            #iterate through query_ids (features)
            query_id_to_hitcount = {}
            query_id_to_predresistant = {}
            
            queryid_to_predsus_sum = {}
            queryid_to_predsus_mean = {}
            queryid_to_predsus_std = {}
            queryid_to_predres_sum = {}
            queryid_to_predres_mean = {}
            queryid_to_predres_std = {}

            for query_id in accession_df['query_id'].unique().tolist():
                query_id_df = accession_df[accession_df['query_id'] == query_id] #this df now only has a single query id for a single accession
                #get number predicted resistant for each query id
                if return_logits is None:
                    num_pred_resistant = query_id_df['pred_res'].map({0: 1, 1: 0}).sum() # need to flip because Susceptible is 1
                    query_id_to_predresistant[f'{query_id}_pred_resistant'] = num_pred_resistant
                elif return_logits == 'sum':
                    num_pred_resistant = query_id_df['pred_res'].sum()
                    query_id_to_predresistant[f'{query_id}_pred_resistant'] = num_pred_resistant
                elif return_logits == 'average':
                    num_pred_resistant = query_id_df['pred_res'].mean()
                    query_id_to_predresistant[f'{query_id}_pred_resistant'] = num_pred_resistant
                elif return_logits == 'all':
                    queryid_to_predsus_sum[f'{query_id}_sus_sum_pred_resistant'] = query_id_df['pred_sus'].sum()
                    queryid_to_predsus_mean[f'{query_id}_sus_mean_pred_resistant'] = query_id_df['pred_sus'].mean()
                    queryid_to_predsus_std[f'{query_id}_sus_std_pred_resistant'] = query_id_df['pred_sus'].std()
                    queryid_to_predres_sum[f'{query_id}_res_sum_pred_resistant'] = query_id_df['pred_res'].sum()
                    queryid_to_predres_mean[f'{query_id}_res_mean_pred_resistant'] = query_id_df['pred_res'].mean()
                    queryid_to_predres_std[f'{query_id}_res_std_pred_resistant'] = query_id_df['pred_res'].std()

                query_id_to_hitcount[f'{query_id}_hit_count'] = query_id_df['hit_count'].iloc[0]
                

            #fill feature lists to match length of header
            hit_count_features_temp = []
            pred_resistant_features_temp = []
            for feature in hit_count_features:
                if feature in query_id_to_hitcount.keys():
                    hit_count_features_temp.append(query_id_to_hitcount[feature])
                else:
                    hit_count_features_temp.append(0)
            for feature in pred_resistant_features:
                if return_logits != 'all':
                    if feature in query_id_to_predresistant.keys():
                        pred_resistant_features_temp.append(query_id_to_predresistant[feature])
                    else:
                        pred_resistant_features_temp.append(0)
                
                else:
                    if feature in queryid_to_predsus_sum.keys():
                        pred_resistant_features_temp.append(queryid_to_predsus_sum[feature])
                    elif feature in queryid_to_predsus_mean.keys():
                        pred_resistant_features_temp.append(queryid_to_predsus_mean[feature])
                    elif feature in queryid_to_predsus_std.keys():
                        pred_resistant_features_temp.append(queryid_to_predsus_std[feature])
                    elif feature in queryid_to_predres_sum.keys():
                        pred_resistant_features_temp.append(queryid_to_predres_sum[feature])
                    elif feature in queryid_to_predres_mean.keys():
                        pred_resistant_features_temp.append(queryid_to_predres_mean[feature])
                    elif feature in queryid_to_predres_std.keys():
                        pred_resistant_features_temp.append(queryid_to_predres_std[feature])
                    else:
                        pred_resistant_features_temp.append(0)


            #write row
            row = [accession, species, antibiotic] + hit_count_features_temp + pred_resistant_features_temp
            if train:
                row.append(accession_df['phenotype'].iloc[0])
            writer.writerow(row)

def eval_consensus(args, train_test):
    """Evaluate DNABERT predictions alone using majority voting,
    saves accession-level DNABERT-only consensus preds to outputs/consensus_preds/run_name.csv
    prints metrics to console and saves to outputs/consensus_preds/consensus_evals.csv
    1 = resistant, 0=susceptible"""

    preds_path = f'{args.base_dir}/dnabert/inference/outputs/preds/{args.run_name}/full/{train_test}/full_preds.csv'
    df = pd.read_csv(preds_path)
    df['pred'] = (df['pred_res'] > df['pred_sus']).astype(int) #1 is resistant, 0 is susceptible (opposite of our usual convention)


    if args.train:
        acc_to_pred = df.groupby('accession').agg({'pred': 'mean'}) #df with accession as index and pred as column
        acc_to_pred['pred'] = acc_to_pred['pred'].map(lambda x: 1 if x > 0.5 else 0) #if more than half are 1 (resistant), then predict resistant (1)
        acc_to_gt = df.groupby('accession').agg({'phenotype': 'mean'})
        acc_to_gt['phenotype'] = acc_to_gt['phenotype'].map({1: 0, 0: 1}) #flip from usual convention to res=1, sus=0
        #print(acc_to_pred)
        #print(acc_to_gt)
        #make output df
        output_df = pd.DataFrame()
        output_df['accession'] = acc_to_pred.index
        output_df['pred_phenotype'] = acc_to_pred['pred'].values
        output_df['gt_phenotype'] = acc_to_gt['phenotype'].values
        output_df.to_csv(f'{args.base_dir}/dnabert/inference/outputs/consensus_preds/{args.run_name}.csv', index=False)

        #evaluate
        num_correct = (acc_to_pred['pred'] == acc_to_gt['phenotype']).sum()
        acc = num_correct / len(acc_to_pred)
        print(f'Accuracy: {acc}')
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(acc_to_gt['phenotype'], acc_to_pred['pred'])
        print(f'Confusion Matrix: {cm}')
        
        #append to evals df
        if not os.path.exists(f'{args.base_dir}/dnabert/inference/outputs/consensus_preds/consensus_evals.csv'):
            evals_df = pd.DataFrame(columns=['run_name', 'accuracy', 'cm'])
        else:
            evals_df = pd.read_csv(f'{args.base_dir}/dnabert/inference/outputs/consensus_preds/consensus_evals.csv')
        evals_df = evals_df._append({'run_name': args.run_name, 'accuracy': acc, 'cm': cm}, ignore_index=True)
        evals_df.to_csv(f'{args.base_dir}/dnabert/inference/outputs/consensus_preds/consensus_evals.csv', index=False)

    return acc, cm

        

def split(full_dataset_path):
    """ opens dev_accs.txt, and test_accs.txt and splits df into train, dev, and test according to those splits """
    dev_accs = [line.rstrip() for line in open('../finetune/data/dev_accs.txt')]
    test_accs = [line.rstrip() for line in open('../finetune/data/test_accs.txt')]

    df = pd.read_csv(full_dataset_path)
    train_df = df[~df['accession'].isin(dev_accs + test_accs)]
    dev_df = df[df['accession'].isin(dev_accs)]
    test_df = df[df['accession'].isin(test_accs)]

    #write to csv
    train_df.to_csv(full_dataset_path.replace('.csv', '_train.csv'), index=False)
    dev_df.to_csv(full_dataset_path.replace('.csv', '_dev.csv'), index=False)
    test_df.to_csv(full_dataset_path.replace('.csv', '_test.csv'), index=False)
    return train_df, dev_df, test_df

def group_dataset(grouping, full_dataset, output_dir):
    #group if not full model
    #only for sequence-based grouping 

    if grouping == 'per_species':
        df = pd.read_csv(full_dataset)
        for species in species_list:
            df_species = df[df['species'] == species_mapping[species]]
            if not os.path.exists(os.path.join(output_dir, species)):
                os.makedirs(os.path.join(output_dir, species))
            df_species.to_csv(os.path.join(output_dir, species, f'{species}_sequence_dataset.csv'), index=False)
            print(f'Wrote {len(df_species)} rows to {species}_sequence_dataset.csv')

    elif grouping == 'per_antibiotic':
        df = pd.read_csv(full_dataset)
        for antibiotic in antibiotic_mapping.keys():
            df_antibiotic = df[df['antibiotic'] == antibiotic_mapping[antibiotic]]
            if not os.path.exists(os.path.join(output_dir, antibiotic)):
                os.makedirs(os.path.join(output_dir, antibiotic))
            df_antibiotic.to_csv(os.path.join(output_dir, antibiotic, '{antibiotic}_sequence_dataset.csv'), index=False)
            print(f'Wrote {len(df_antibiotic)} rows to {antibiotic}_sequence_dataset.csv')



def main():

    #parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_format', type=str, choices=['random_forest', 'consensus'], default='random_forest')
    parser.add_argument('--dataset_dir', type=str, help='Path to directory containing datasets formatted by data_pipeline/pipeline.py, if full model should be direct path to dataset', default=None)
    parser.add_argument('--model_path', type=str, help='Path to DNABERT model, if full model, should be direct path to model, if per_species, should be base_dir where base_dir/{species}/best is each model', default=None)
    parser.add_argument('--run_name', type=str, help='Name of run', default='unspecified_model')
    parser.add_argument('--grouping', type=str, help='grouping for DNABERT model (not the output datasets, these will always be per-species)', choices=['full', 'per_species', 'per_antibiotic'], default='per_species')
    parser.add_argument('--sig_seqs_dir', type=str, help='Path to directory containing significant sequences from DBGWAS (for IDs only here)', default='/gpfs/scratch/jvaska/CAMDA_AMR/AMR_v2/data_pipeline/data/dbgwas/p0.05/per_species')
    parser.add_argument('--train', type=bool, help='will add ground truth labels', default=False)
    parser.add_argument('--split', type=bool, help='will look for dev_accs.txt and test_accs.txt to split each final dataset into train/test/dev according to split that dnabert was trained with', default=False)
    parser.add_argument('--metadata_dir', default='/gpfs/scratch/jvaska/CAMDA_AMR/AMR_v2/data_pipeline/data/metadata')
    parser.add_argument('--base_dir', type=str, help='base directory that this script is in', default='/gpfs/scratch/jvaska/CAMDA_AMR/AMR_v2')
    parser.add_argument('--return_logits', type=str, choices=['sum', 'average', 'std', 'all', None], help='Dont take argmax while filling out pred_phenos csv. Input to RF will be sum of resistant probabilities', default=None)
    args = parser.parse_args()
    print(f'Arguments: {args}')
    train_test = 'train' if args.train else 'test'

    #open species_to_accessions.json
    with open(os.path.join(args.metadata_dir, f'{train_test}_species_to_accession.json')) as f:
        species_to_accessions = json.load(f)
        
    

    
    #loop through dataset depending on grouping
    if args.grouping == 'per_species':
        for species in species_list:
            #get model
            model_path = os.path.join(args.model_path, species, 'best') #WARNING - if we change the model architecture, huggingface stores a model called 'best' in its cache, and won't reload it when this is called again, so make sure we clear cache or change the best model dirname from 'best'
            model, tokenizer = load_model(model_path)

            dataset_path = os.path.join(args.dataset_dir, species, f'{species}_sequence_dataset.csv') #load sequence dataset
            preds_path = f'{args.base_dir}/dnabert/inference/outputs/preds/{args.run_name}/per_species/{train_test}/{species}_preds.csv'
            if not os.path.exists(preds_path):
                os.makedirs(os.path.dirname(preds_path), exist_ok=True)
                print(f'Running DNABERT inference for {species}')
                inference(dataset_path, preds_path, model, tokenizer, args.return_logits)
            if args.output_format == 'random_forest':
                output_path = f'{args.base_dir}/dnabert/inference/outputs/rf_datasets/{args.run_name}/per_species/{train_test}/{species}/{species}_full_rf_dataset.csv'
                if not os.path.exists(os.path.dirname(output_path)):
                    os.makedirs(os.path.dirname(output_path))
                print(f'Getting RF dataset for {species}')
                get_rf_dataset(preds_path, output_path, args.train, os.path.join(args.sig_seqs_dir, f'{species}_sig_sequences.fasta'), accessions = species_to_accessions[species], return_logits=args.return_logits)
            elif args.output_format == 'consensus':
                pass

            if args.split:
                split(output_path)

    elif args.grouping == 'per_antibiotic':
        for antibiotic in antibiotic_mapping.keys():

            model_path = os.path.join(args.model_path, antibiotic, 'best') #WARNING - if we change the model architecture, huggingface stores a model called 'best' in its cache, and won't reload it when this is called again, so make sure we clear cache or change the best model dirname from 'best'
            model, tokenizer = load_model(model_path)
            dataset_path = os.path.join(args.dataset_dir, antibiotic, f'{antibiotic}_sequence_dataset.csv') #load sequence dataset
            preds_path = f'{args.base_dir}/dnabert/inference/outputs/preds/{args.run_name}/per_antibiotic/{train_test}/{antibiotic}_preds.csv'
            if not os.path.exists(preds_path):
                os.makedirs(os.path.dirname(preds_path), exist_ok=True)
                print(f'Running DNABERT inference for {antibiotic}')
                inference(dataset_path, preds_path, model, tokenizer, args.return_logits)
            if args.output_format == 'random_forest':
                for species in antibiotic_to_species[antibiotic]:
                    output_path = f'{args.base_dir}/dnabert/inference/outputs/rf_datasets/{args.run_name}/per_antibiotic/{train_test}/{species}/{species}_full_rf_dataset.csv'
                    if not os.path.exists(os.path.dirname(output_path)):
                        os.makedirs(os.path.dirname(output_path))
                    print(f'Getting RF dataset for {species}')
                    #we need to filter preds_path to contain only sequences from this species
                    get_rf_dataset(preds_path, output_path, args.train, os.path.join(args.sig_seqs_dir, f'{species}_sig_sequences.fasta'), accessions = species_to_accessions[species], return_logits=args.return_logits)
                    if args.split:
                        split(output_path)
            elif args.output_format == 'consensus':
                pass

    elif args.grouping == 'full':
        model, tokenizer = load_model(args.model_path)
        preds_path = f'{args.base_dir}/dnabert/inference/outputs/preds/{args.run_name}/full/{train_test}/full_preds.csv'
        if not os.path.exists(preds_path):
            os.makedirs(os.path.dirname(preds_path), exist_ok=True)
            print(f'Running DNABERT inference for full model')
            inference(args.dataset_dir, preds_path, model, tokenizer, args.return_logits)
        
        if args.output_format == 'random_forest':
            for species in species_list:
                output_path = f'{args.base_dir}/dnabert/inference/outputs/rf_datasets/{args.run_name}/full/{train_test}/{species}/{species}_full_rf_dataset.csv'
                if not os.path.exists(os.path.dirname(output_path)):
                    os.makedirs(os.path.dirname(output_path))
                print(f'Getting RF dataset for {species}')
                get_rf_dataset(preds_path, output_path, args.train, os.path.join(args.sig_seqs_dir, f'{species}_sig_sequences.fasta'), accessions = species_to_accessions[species], return_logits=args.return_logits)
                if args.split:
                    split(output_path)

        elif args.output_format == 'consensus':
            acc, cm = eval_consensus(args, train_test)
            print(f'Accuracy: {acc}')
            print(f'Confusion Matrix: {cm}')


if __name__ == "__main__":
    main()