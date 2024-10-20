import os
from pathlib import Path
import pandas as pd
import tarfile
from collections import defaultdict
import random

def extract_images_for_gene_ko_and_controls(metadata_file, data_directory, gene, control_sample_rate=0.1):
    # Load metadata
    metadata = pd.read_csv(metadata_file)

    # Create a dictionary to store filenames by treatment, experiment, plate, and well
    filenames_by_treatment = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))

    # Function to process and store filenames
    def process_row(row, treatment):
        experiment_name = row['experiment_name']
        plate = f"Plate{row['plate']}"
        well = row['well_id']
        address = row['address']
        
        for i in range(1, 7):
            filename = f'{address}_s1_w{i}.png'
            filenames_by_treatment[treatment][experiment_name][plate][well].append(filename)

    # Process gene knockouts and identify relevant plates
    ko_plates = set()
    gene_ko_metadata = metadata[metadata['gene'] == gene]
    for _, row in gene_ko_metadata.iterrows():
        process_row(row, f"{gene}_KO")
        ko_plates.add((row['experiment_name'], row['plate']))

    # Process EMPTY_controls only for relevant plates
    for (experiment_name, plate) in ko_plates:
        plate_df = metadata[(metadata['experiment_name'] == experiment_name) & 
                            (metadata['plate'] == plate) &
                            (metadata['treatment'] == 'EMPTY_control')]
        
        sampled_controls = plate_df.sample(frac=control_sample_rate, random_state=42)
        for _, row in sampled_controls.iterrows():
            process_row(row, 'EMPTY_control')

    # Extract files from tar archives
    for treatment, experiments in filenames_by_treatment.items():
        for experiment_name, plates in experiments.items():
            output_dir = Path(f'Gene_KO_and_controls/{treatment}')
            output_dir.mkdir(parents=True, exist_ok=True)

            for plate, wells in plates.items():
                tar_file = Path(data_directory) / f'{plate}.tar'
                
                if not tar_file.exists():
                    print(f'Tar file not found: {tar_file}')
                    continue
                
                with tarfile.open(tar_file, 'r') as tar:
                    for well, filenames in wells.items():
                        well_dir = output_dir / f'{experiment_name}_{plate}_{well}'
                        well_dir.mkdir(exist_ok=True)
                        
                        for filename in filenames:
                            try:
                                member = tar.getmember(filename)
                                member.name = os.path.basename(member.name)
                                tar.extract(member, path=well_dir)
                                print(f'Extracted: {filename} to {well_dir}')
                            except KeyError:
                                print(f'File not found in tar: {filename}')

# Usage
metadata_file = 'metadata_rxrx3.csv'
data_directory = '/mnt/data/rxrx3/images/gene-081'
gene_of_interest = 'EGFR'  # Replace with the gene you're interested in

extract_images_for_gene_ko_and_controls(metadata_file, data_directory, gene_of_interest)