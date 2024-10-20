import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import pdist, squareform

def load_data(file_path):
    """Load the phenoprint data"""
    return pd.read_csv(file_path)

def calculate_pairwise_distances(data):
    """Calculate pairwise distances between phenoprints"""
    return pdist(data.drop(['well', 'position'], axis=1), metric='euclidean')

def map_distances_to_positions(distances, data):
    """Map distances to well positions"""
    dist_matrix = squareform(distances)
    n = len(data)
    position_distances = []
    for i in range(n):
        for j in range(i+1, n):
            position_distances.append({
                'distance': dist_matrix[i, j],
                'well1': data.iloc[i]['well'],
                'position1': data.iloc[i]['position'],
                'well2': data.iloc[j]['well'],
                'position2': data.iloc[j]['position']
            })
    return pd.DataFrame(position_distances)

def analyze_position_effects(position_distances):
    """Analyze the effect of well position on distances"""
    # Group by position combinations
    position_effects = position_distances.groupby(['position1', 'position2'])['distance'].mean().reset_index()
    
    # heatmap of position effects
    pivot_positions = position_effects.pivot(index='position1', columns='position2', values='distance')
    plt.figure(figsize=(10, 8))
    sns.heatmap(pivot_positions, annot=True, cmap='YlOrRd')
    plt.title('Average Distance by Image Position Combination')
    plt.savefig('position_effects_heatmap.png')
    plt.close()

def analyze_well_effects(position_distances):
    """Analyze the effect of well location on distances"""
    position_distances['row1'] = position_distances['well1'].str[0]
    position_distances['col1'] = position_distances['well1'].str[1:]
    position_distances['row2'] = position_distances['well2'].str[0]
    position_distances['col2'] = position_distances['well2'].str[1:]
    
    # average distance for each well
    well_effects = position_distances.groupby('well1')['distance'].mean().reset_index()
    well_effects['row'] = well_effects['well1'].str[0]
    well_effects['col'] = well_effects['well1'].str[1:].astype(int)
    
    #  a heatmap of well effects
    pivot_wells = well_effects.pivot(index='row', columns='col', values='distance')
    plt.figure(figsize=(15, 10))
    sns.heatmap(pivot_wells, annot=True, cmap='YlOrRd')
    plt.title('Average Distance by Well Location')
    plt.savefig('well_effects_heatmap.png')
    plt.close()

def identify_outliers(position_distances, threshold=2):
    """Identify outlier cells based on their average distance"""
    cell_avg_distances = position_distances.groupby(['well1', 'position1'])['distance'].mean().reset_index()
    global_mean = cell_avg_distances['distance'].mean()
    global_std = cell_avg_distances['distance'].std()
    outliers = cell_avg_distances[np.abs(cell_avg_distances['distance'] - global_mean) > threshold * global_std]
    return outliers

def main():
    data = load_data('phenoprint_data.csv')
    
    distances = calculate_pairwise_distances(data)
    
    position_distances = map_distances_to_positions(distances, data)
    
    analyze_position_effects(position_distances)
    
    analyze_well_effects(position_distances)
    
    outliers = identify_outliers(position_distances)
    print("Outlier cells:")
    print(outliers)
    
    outliers.to_csv('outlier_cells.csv', index=False)

if __name__ == '__main__':
    main()
