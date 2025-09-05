#!/bin/bash

DBGWAS_DIR=/gpfs/scratch/jvaska/CAMDA_AMR/AMR_v2/data_pipeline/data/dbgwas/p0.05/per_species

# Check if DBGWAS_DIR is set and exists
if [ -z "$DBGWAS_DIR" ] || [ ! -d "$DBGWAS_DIR" ]; then
    echo "Error: DBGWAS_DIR is not set or directory does not exist"
    echo "Please update the DBGWAS_DIR variable in this script"
    exit 1
fi

# Function to get species ID from species name
get_species_id() {
    local species_name=$1
    case $species_name in
        "klebsiella_pneumoniae") echo "0" ;;
        "streptococcus_pneumoniae") echo "1" ;;
        "escherichia_coli") echo "2" ;;
        "campylobacter_jejuni") echo "3" ;;
        "salmonella_enterica") echo "4" ;;
        "neisseria_gonorrhoeae") echo "5" ;;
        "staphylococcus_aureus") echo "6" ;;
        "pseudomonas_aeruginosa") echo "7" ;;
        "acinetobacter_baumannii") echo "8" ;;
        *) echo "unknown" ;;
    esac
}

# Process each .fasta file in DBGWAS_DIR
for fasta_file in "$DBGWAS_DIR"/*_sig_sequences.fasta; do
    if [ -f "$fasta_file" ]; then
        echo "Processing: $fasta_file"
        
        # Extract species name from filename (remove _sig_sequences.fasta suffix)
        filename=$(basename "$fasta_file")
        species_name="${filename%_sig_sequences.fasta}"
        
        # Get species ID
        species_id=$(get_species_id "$species_name")
        
        if [ "$species_id" = "unknown" ]; then
            echo "Warning: Unknown species '$species_name' in file $fasta_file"
            continue
        fi
        
        # Create temporary file
        temp_file=$(mktemp)
        sequence_id=0
        
        # Process each line in the fasta file
        while IFS= read -r line; do
            if [[ $line =~ ^\> ]]; then
                # This is a header line, rename the ID
                new_id="${species_id}_${sequence_id}"
                echo ">${new_id}" >> "$temp_file"
                sequence_id=$((sequence_id + 1))
            else
                # This is a sequence line, copy as-is
                echo "$line" >> "$temp_file"
            fi
        done < "$fasta_file"
        
        # Replace original file with renamed version
        mv "$temp_file" "$fasta_file"
        echo "Renamed $((sequence_id)) sequences in $fasta_file"
    fi
done

echo "All .fasta files processed successfully!"
