#Helps visualize how cell populations respond to different drugs over time, 
# which essentially can show differences in drug mechanisms and efficacy.

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

def load_time_series_data(file_path):
    data = pd.read_csv(file_path)
    return data

def preprocess_data(data):
    features = data.drop(['time', 'drug', 'concentration'], axis=1)
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    return scaled_features, data[['time', 'drug', 'concentration']]

def perform_pca(scaled_features):
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(scaled_features)
    return pca_result

def plot_trajectories(pca_result, metadata):
    plt.figure(figsize=(12, 8))
    for drug in metadata['drug'].unique():
        drug_data = metadata[metadata['drug'] == drug]
        plt.plot(pca_result[drug_data.index, 0], pca_result[drug_data.index, 1], 
                 '-o', label=drug)
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('Drug Response Trajectories')
    plt.legend()
    plt.savefig('drug_response_trajectories.png')

def analyze_drug_responses(file_path):
    data = load_time_series_data(file_path)
    scaled_features, metadata = preprocess_data(data)
    pca_result = perform_pca(scaled_features)
    plot_trajectories(pca_result, metadata)

if __name__ == "__main__":
    analyze_drug_responses('time_series_drug_response.csv')
