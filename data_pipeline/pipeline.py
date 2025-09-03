"""
This script is to be used on the CAMDA AMR Prediction Challenge 2025 train and test datasets
This script takes as input a path to a directory of assemblies (.fasta files) and returns a dataset for specified model

Key Options:
    - "model type": "sequence-based" or "matrix-based"
        - "sequence-based": e.g. DNABERT, model takes as input sequence (+ num_hits and optionally species/antibiotic)
        - "matrix-based": e.g. Random Forest, model takes as input matrix of features (gene counts and optionally species/antibiotic)

    - "grouping": "full", "per_species", or "per_antibiotic"
        - "full": returns single dataset for all assemblies
        - "per-species": returns one dataset per species (9 datasets)
        - "per-antibiotic": returns one dataset per antibiotic (4 datasets)

    - "split": path to directory containing train/test/dev accession lists (train.txt, test.txt, dev.txt), default is None
        - TODO: NOT YET IMPLEMENTED
        - ***this is for split of the TRAIN SET for evaluation before submission*** to train on entire train set, dont specify
        - if not specified, will return single dataset

Paths to be specified:
    - "train_assemblies": path to directory containing all train assemblies (.fasta files)
        - if not specified, will only generate test set
    - "test_assemblies": path to directory containing all test assemblies
        - if not specified, will only generate train set
    - "train_metadata": path to CSV containing accessions, ground truth labels, genus, and species columns
        - must specify if "train_assemblies" is specified
        - as downloaded from CAMDA website
    - "test_template": path to testing template TSV
        - must specify if "test_assemblies" is specified
        - as downloaded from CAMDA website
    - "perspecies_dbgwas_dir": path to directory containing significant sequences from DBGWAS 
        - 9 fasta files: {species}_sig_sequences.fasta
    - "output_dir": base directory for output datasets, default is ./datasets

Outputs/Next Steps:
    - <output_dir>/<model_type>/<grouping>/<train/test (official CAMDA split)>/<[full]/[species]/[antibiotic].csv>
    - sequence-based datasets should be sent to dnabert directory for finetuning/inference
    - matrix-based datasets should be sent to rf or xgboost directory for finetuning/inference
    - if "split" is specified, <train>/<test>/... can be passed to (TODO) evaluation script for accession-level evaluation before submission



Additional parameters listed below, but should remain relatively constant and only be modified once at a time for consistency


"""


## parameters: keep these relatively constant for consistency

#universal parameters for blast
BLAST_IDENTITY = 85
LEAKAGE = True #should only set to False to prevent data leakage when running an abalation study on spcecies/antibiotic features on full/per-antibiotic models
DBGWAS_SIG_LEVEL = str(0.05) #THIS IS JUST TO FIND APPROPRIATE FILES, we do NOT filter sequences in this script, this should be done in DBGWAS script
MAX_TARGET_SEQS = 10 #for BLAST


#sequence-based parameters
SEQ_LENGTH = 1000
MIN_SEQS = 10
TOPUP = False

#matrix-based parameters
BINARY_FEATURES = False


## global mappings:
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

antibiotic_list = ['GEN', 'ERY', 'CAZ', 'TET']

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

antibiotic_to_species = {
    'GEN': ['klebsiella_pneumoniae', 'escherichia_coli', 'salmonella_enterica'],
    'ERY': ['streptococcus_pneumoniae', 'staphylococcus_aureus'],
    'CAZ': ['pseudomonas_aeruginosa', 'acinetobacter_baumannii'],
    'TET': ['campylobacter_jejuni', 'neisseria_gonorrhoeae'],
}

# index mappings:
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

antibiotic_mapping = {'GEN': 0, 
    'ERY': 1,
    'CAZ': 2,
    'TET': 3,
}

# imports
import os
import argparse




# methods:


def group_sig_seqs(grouping, output_dir, perspecies_dir): #NOT YET TESTED
    #if grouping is full, concat all species files to full_sig_sequences.fasta
    if grouping == 'full':
        with open(os.path.join(output_dir, 'full_sig_sequences.fasta'), 'w') as output_file:
        for species in species_list:
            with open(os.path.join(perspecies_dir, f'{species}_sig_sequences.fasta'), 'r') as f:
                for line in f:
                    if line.startswith('>'):
                        output_file.write(line)

    #if grouping is per_antibiotic, concat all antibiotic files to TET_sig_sequences.fasta, GEN_sig_sequences.fasta, etc. according to antibiotic_to_species
    elif grouping == 'per_antibiotic':
        for antibiotic in antibiotic_list:
            with open(os.path.join(output_dir, f'{antibiotic}_sig_sequences.fasta'), 'w') as output_file:
                for species in antibiotic_to_species[antibiotic]:
                    with open(os.path.join(perspecies_dir, f'{species}_sig_sequences.fasta'), 'r') as f:
                        for line in f:
                            if line.startswith('>'):
                                output_file.write(line)

    elif grouping == 'per_species':
        return

    else:
        raise ValueError(f'Invalid grouping: {grouping}')


def get_hits(accessions_list, assemblies_dir, species_to_accessions, dbgwas_dir, leakage, grouping):
    #runs BLAST to align sequences in dbgwas_dir to assemblies
    #if leakage is True (default), aligns assemblies to their species-specific features
    #if leakage is False, aligns to all sig seqs (if grouping is full), or to all sig seqs in antibiotic grouping (if grouping is per_antibiotic)

    if not os.path.exists('work/blast/dbs'):
        os.makedirs('work/blast/dbs')
    if not os.path.exists('work/blast/hits'):
        os.makedirs('work/blast/hits')

    if leakage or grouping == 'per_species':
        for species in species_list:
            accessions = species_to_accessions[species] and accessions_list
            sig_seqs_path = os.path.join(dbgwas_dir, f'{species}_sig_sequences.fasta')
            hits_subdir = 'species_specific'
            if not os.path.exists(os.path.join('work/blast/hits', hits_subdir)):
                os.makedirs(os.path.join('work/blast/hits', hits_subdir))

            run_blast()
                
    else:
        if grouping == 'full':
            sig_seqs_path = os.path.join(dbgwas_dir, 'full_sig_sequences.fasta')
            accessions = accessions_list
            hits_subdir = 'full'
            if not os.path.exists(os.path.join('work/blast/hits', hits_subdir)):
                os.makedirs(os.path.join('work/blast/hits', hits_subdir))

            run_blast()

        elif grouping == 'per_antibiotic':
            for antibiotic in antibiotic_list:
                antibiotic_species = antibiotic_to_species[antibiotic]
                accessions = []
                for species in antibiotic_species:
                    accessions.extend(species_to_accessions[species])
                sig_seqs_path = os.path.join(dbgwas_dir, f'{antibiotic}_sig_sequences.fasta')
                hits_subdir = 'per_antibiotic'
                if not os.path.exists(os.path.join('work/blast/hits', hits_subdir)):
                    os.makedirs(os.path.join('work/blast/hits', hits_subdir))

                run_blast()

    def run_blast():
        for accession in accessions:
                assembly_path = os.path.join(assemblies_dir, f'{accession}.fasta')
                db_dir = os.path.join('work', 'blast', 'dbs', accession)
                if not os.path.exists(db_dir):
                    cmd = f'makeblastdb -in {assembly_path} -dbtype nucl -out {db_dir}'
                    subprocess.run(cmd, shell=True)
                hits_file = os.path.join('work', 'blast', 'hits', hits_subdir, f'{accession}_hits.tsv')
                if not os.path.exists(hits_file):
                    cmd = (f'blastn -query {sig_seqs_path} -db {db_dir} '
                        f'-max_target_seqs {MAX_TARGET_SEQS} -outfmt "6 qseqid sseqid pident length '
                        f'qstart qend sstart send sstrand bitscore" -perc_identity {BLAST_IDENTITY} '
                        f'-out {hits_file}')
                    subprocess.run(cmd, shell=True)

       

    
def main():


    # parse arguments
    parser = argparse.ArgumentParser(description='CAMDA AMR Prediction Challenge 2025 Data Pipeline')
    parser.add_argument('--train_assemblies', type=str, help='Path to directory containing all train assemblies (.fasta files)', default=None)
    parser.add_argument('--test_assemblies', type=str, help='Path to directory containing all test assemblies (.fasta files)', default=None)
    parser.add_argument('--train_metadata', type=str, help='Path to CSV containing accessions, ground truth labels, genus, and species columns', default=None)
    parser.add_argument('--test_template', type=str, help='Path to testing template TSV', default=None)
    parser.add_argument('--output_dir', type=str, help='Base directory for output datasets', default='./datasets')
    parser.add_argument('--model_type', type=str, help='Model type', choices=['sequence-based', 'matrix-based'], default='sequence-based')
    parser.add_argument('--grouping', type=str, help='Grouping', choices=['full', 'per_species', 'per_antibiotic'], default='full')
    parser.add_argument('--split', type=str, help='Path to directory containing train/test/dev accession lists (train.txt, test.txt, dev.txt)', default=None)
    parser.add_argument('--perspecies_dbgwas_dir', type=str, help='Path to directory containing significant sequences from DBGWAS', default=f'./data/dbgwas/p{DBGWAS_SIG_LEVEL}/per_species')
    args = parser.parse_args()


    # check arguments
    print(f'Arguments: {args}')
    if args.train_assemblies is None and args.test_assemblies is None:
        raise ValueError('Either train_assemblies or test_assemblies must be specified')
    if args.train_metadata is None and args.train_assemblies is not None:
        raise ValueError('train_metadata must be specified if train_assemblies is specified')
    if args.test_template is None and args.test_assemblies is not None:
        raise ValueError('test_template must be specified if test_assemblies is specified')
    train = False
    test = False
    if args.train_assemblies:
        print(f'Getting train set')
        train = True
    if args.test_assemblies:
        print(f'Getting test set')
        test = True

    # get base output directory
    base_output_dir = os.path.join(args.output_dir, args.model_type, args.grouping) #from here add /train or /test (official CAMDA split)
    if train:
        train_output_dir = os.path.join(base_output_dir, 'train')
    if test:
        test_output_dir = os.path.join(base_output_dir, 'test')

    # parse metadata
    #read dataframes, create genus_species column from genus and species columns
    #create dictionaty mapping genus_species to list of accessions
    #create accession lists for train and test
    if train:
        train_metadata = pd.read_csv(args.train_metadata)
        train_metadata['genus_species'] = train_metadata['genus'] + '_' + train_metadata['species']
        train_accessions = train_metadata['accession'].tolist()
        train_species_to_accessions = train_metadata.groupby('genus_species')['accession'].apply(list).to_dict()

    if test:
        test_metadata = pd.read_csv(args.test_template)
        test_metadata['genus_species'] = test_metadata['genus'] + '_' + test_metadata['species']
        test_accessions = test_metadata['accession'].tolist()
        test_species_to_accessions = test_metadata.groupby('genus_species')['accession'].apply(list).to_dict()



    ## start pipeline 

    # if running an abalation study on species/antibiotic features, prevent species/antibiotic data leakage by grouping significant sequences by species/antibiotic
    if not LEAKAGE:
        #group DBGWAS significant sequencesfasta files:
        #for full model, groups all to ./data/dbgwas/p{threshold}/full/full_sig_sequences.fasta
        #for per-antibiotic model, groups all to ./data/dbgwas/p{threshold}/per_antibiotic/TET_sig_sequences.fasta, GEN_sig_sequences.fasta, etc. according to antibiotic_to_species
        #for per-species model, does not group (input should be grouped by species, this is how DBGWAS is run), files stay in args.perspecies_dbgwas_dir
        dbgwas_dir = os.path.join('data', 'dbgwas', f'p{DBGWAS_SIG_LEVEL}', args.grouping)
        group_sig_seqs(args.grouping, dbgwas_dir, args.perspecies_dbgwas_dir)
    else:
        dbgwas_dir = args.perspecies_dbgwas_dir

    # run BLAST
    #generates {accession}_hits.tsv files
    #aligns sequences in dbgwas_dir to assemblies
    #normally aligns assemblies to their species-specific features, but if LEAKAGE is False, takes args.grouping and aligns to all sequences in dbgwas_dir
    #output {accession}_hits.tsv files in work/blast/hits/species_specific/{accession}_hits.tsv by default, unless LEAKAGE is False
    if train:
        get_hits(accessions_list=train_accessions, assemblies_dir=args.train_assemblies, species_to_accessions=train_species_to_accessions, dbgwas_dir=dbgwas_dir, leakage=LEAKAGE, grouping=args.grouping)
    if test:
        get_hits(accessions_list=test_accessions, assemblies_dir=args.test_assemblies, species_to_accessions=test_species_to_accessions, dbgwas_dir=dbgwas_dir, leakage=LEAKAGE, grouping=args.grouping)

    # get datasets
    #run generate_matrix() for matrix-based models and generate_sequence_dataset() for sequence-based models
    #


    



if __name__ == "__main__":
    main()