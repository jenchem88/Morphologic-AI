import os
from pathlib import Path
import pandas as pd
import tarfile
from collections import defaultdict

def extract_images_for_egfr_inhibitors(metadata_file, data_volume1, data_volume2):
    # Load metadata
    metadata = pd.read_csv(metadata_file)

    # List of EGFR inhibitors
    egfr_inhibitors = [
        'Neratinib', 'Gefitinib', 'Lapatinib', 
        'afatinib', 'Icotinib', 'Vandetanib', 
        'Olmutinib', 'AZD9291', 'Erlotinib', 
        'OSI-420', 'Dacomitinib'
    ]

    # Create a dictionary to store filenames by treatment, experiment, plate, and well
    filenames_by_treatment = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))

    # Populate the dictionary
    for treatment in egfr_inhibitors:
        treatment_metadata = metadata[metadata['treatment'] == treatment]
        
        for _, row in treatment_metadata.iterrows():
            experiment_name = row['experiment_name']
            plate = f"Plate{row['plate']}"
            well = row['well_id']
            address = row['address']
            concentration = row['concentration']
            
            for i in range(1, 7):
                filename = f'{address}_s1_w{i}.png'
                filenames_by_treatment[treatment][experiment_name][plate][well].append((filename, concentration))

    # Extract files from tar archives
    for treatment, experiments in filenames_by_treatment.items():
        for experiment_name, plates in experiments.items():
            # Determine the correct image folder based on the experiment name
            if experiment_name == 'compound-002':
                image_folder = Path(data_volume1) / 'rxrx3/images' / experiment_name
            else:
                image_folder = Path(data_volume2) / experiment_name

            output_dir = Path(f'EGFR_inhibitors/{treatment}')
            output_dir.mkdir(parents=True, exist_ok=True)

            for plate, wells in plates.items():
                tar_file = image_folder / f'{plate}.tar'
                
                if not tar_file.exists():
                    print(f'Tar file not found: {tar_file}')
                    continue
                
                with tarfile.open(tar_file, 'r') as tar:
                    for well, filenames in wells.items():
                        well_dir = output_dir / f'{experiment_name}_{plate}_{well}_conc{filenames[0][1]}'
                        well_dir.mkdir(exist_ok=True)
                        
                        for filename, _ in filenames:
                            try:
                                member = tar.getmember(filename)
                                member.name = os.path.basename(member.name)
                                tar.extract(member, path=well_dir)
                                print(f'Extracted: {filename} to {well_dir}')
                            except KeyError:
                                print(f'File not found in tar: {filename}')

# Usage
metadata_file = 'metadata_rxrx3.csv'
data_volume1 = '/mnt/data'
data_volume2 = '/mnt/data2'

extract_images_for_egfr_inhibitors(metadata_file, data_volume1, data_volume2)