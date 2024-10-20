import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from kneed import KneeLocator
import argparse
import logging

def load_and_preprocess_data(file_path):
    data = pd.read_csv(file_path)
    features = data.drop(['drug_name', 'concentration'], axis=1)
    metadata = data[['drug_name', 'concentration']]
    
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    
    return scaled_features, metadata

def perform_pca(data, variance_threshold=0.95):
    pca = PCA(n_components=variance_threshold, svd_solver='full')
    pca_result = pca.fit_transform(data)
    n_components = pca.n_components_
    logging.info(f"Number of PCA components retained: {n_components}")
    return pca_result, pca

def determine_optimal_clusters(data, max_clusters=15):
    inertias = []
    for k in range(1, max_clusters + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(data)
        inertias.append(kmeans.inertia_)
    
    kl = KneeLocator(range(1, max_clusters + 1), inertias, curve="convex", direction="decreasing")
    optimal_clusters = kl.elbow
    logging.info(f"Optimal number of clusters: {optimal_clusters}")
    return optimal_clusters

def perform_clustering(data, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(data)
    return cluster_labels

def visualize_clusters(pca_result, cluster_labels):
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(pca_result[:, 0], pca_result[:, 1], c=cluster_labels, cmap='viridis')
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.title('Drug Clusters in PCA Space')
    plt.colorbar(scatter, label='Cluster Label')
    plt.savefig('cluster_visualization.png')
    plt.close()

def generate_cluster_heatmap(original_data, cluster_labels):
    clustered_data = pd.DataFrame(original_data)
    clustered_data['Cluster'] = cluster_labels
    cluster_means = clustered_data.groupby('Cluster').mean()
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(cluster_means, cmap='YlGnBu', annot=False)
    plt.title('Cluster Characteristics Heatmap')
    plt.savefig('cluster_heatmap.png')
    plt.close()

def analyze_clusters(metadata, cluster_labels, pca, original_features):
    results = pd.DataFrame({
        'drug_name': metadata['drug_name'],
        'concentration': metadata['concentration'],
        'cluster': cluster_labels
    })
    
    cluster_summary = {}
    for cluster in range(max(cluster_labels) + 1):
        cluster_drugs = results[results['cluster'] == cluster]['drug_name'].unique()
        cluster_summary[cluster] = cluster_drugs.tolist()
    
    feature_importance = pd.DataFrame(pca.components_.T, columns=[f'PC{i+1}' for i in range(pca.n_components_)], index=original_features.columns)
    top_features = feature_importance.abs().sum().sort_values(ascending=False).head(10).index.tolist()
    
    return cluster_summary, top_features

def generate_report(cluster_summary, top_features):
    with open('cluster_analysis_report.txt', 'w') as f:
        f.write("Cluster Analysis Report\n")
        f.write("=======================\n\n")
        for cluster, drugs in cluster_summary.items():
            f.write(f"Cluster {cluster}:\n")
            for drug in drugs:
                f.write(f"  - {drug}\n")
            f.write("\n")
        
        f.write("Top 10 Important Features:\n")
        for feature in top_features:
            f.write(f"  - {feature}\n")

def main(file_path):
    logging.basicConfig(level=logging.INFO)
    
    # Load and preprocess data
    scaled_features, metadata = load_and_preprocess_data(file_path)
    
    # Perform PCA
    pca_result, pca = perform_pca(scaled_features)
    
    # Determine optimal number of clusters
    optimal_clusters = determine_optimal_clusters(pca_result)
    
    # Perform clustering
    cluster_labels = perform_clustering(pca_result, optimal_clusters)
    
    # Visualize clusters
    visualize_clusters(pca_result, cluster_labels)
    
    # Generate cluster heatmap
    generate_cluster_heatmap(scaled_features, cluster_labels)
    
    # Analyze clusters
    cluster_summary, top_features = analyze_clusters(metadata, cluster_labels, pca, pd.DataFrame(scaled_features))
    
    # Generate report
    generate_report(cluster_summary, top_features)
    
    logging.info("Analysis complete. Check output files for results.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PCA-based Clustering Tool for Drug Response Analysis")
    parser.add_argument("file_path", help="Path to the CSV file containing drug response data")
    args = parser.parse_args()
    main(args.file_path)
