import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from skimage import io, filters, measure, segmentation
from scipy import ndimage, stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import networkx as nx
from tqdm import tqdm


def load_and_preprocess_images(file_paths):
    """Load and preprocess multi-channel images."""
    channels = [io.imread(path) for path in file_paths]
    normalized_channels = [exposure.rescale_intensity(ch, out_range=(0, 1)) for ch in channels]
    return normalized_channels

def segment_composite(channels):
    """Segment the composite image."""
    composite = np.max(channels, axis=0)
    thresh = filters.threshold_otsu(composite)
    return composite > thresh

def extract_features(channels, mask):
    """Extract features from segmented multi-channel image."""
    labeled_mask = measure.label(mask)
    props = measure.regionprops(labeled_mask)
    
    features = []
    for region in props:
        cell_features = {
            'area': region.area,
            'perimeter': region.perimeter,
            'eccentricity': region.eccentricity,
            'solidity': region.solidity,
        }
        
        for i, channel in enumerate(channels):
            channel_intensity = channel[region.coords[:, 0], region.coords[:, 1]]
            cell_features[f'channel_{i}_mean_intensity'] = np.mean(channel_intensity)
            cell_features[f'channel_{i}_std_intensity'] = np.std(channel_intensity)
        
        # Specific organelle features
        nucleus_channel, mito_channel, golgi_channel = 0, 1, 2  # Adjust based on actual channel order
        cell_features['nucleus_centroid'] = ndimage.center_of_mass(region.intensity_image[:,:,nucleus_channel])
        cell_features['mito_centroid'] = ndimage.center_of_mass(region.intensity_image[:,:,mito_channel])
        cell_features['golgi_centroid'] = ndimage.center_of_mass(region.intensity_image[:,:,golgi_channel])
        
        # RNA distribution (assuming RNA is channel 3 - please adjust if not)
        rna_channel = 3
        rna_intensity = channels[rna_channel][region.coords[:, 0], region.coords[:, 1]]
        cell_features['rna_mean'] = np.mean(rna_intensity)
        cell_features['rna_std'] = np.std(rna_intensity)
        cell_features['rna_cv'] = cell_features['rna_std'] / cell_features['rna_mean'] if cell_features['rna_mean'] > 0 else 0
        
        # Actin structure (assuming actin is channel 4 - please adjust if not)
        actin_channel = 4
        actin_binary = channels[actin_channel][region.coords[:, 0], region.coords[:, 1]] > np.mean(channels[actin_channel])
        actin_skeleton = ndimage.skeletonize(actin_binary)
        cell_features['actin_total_length'] = np.sum(actin_skeleton)
        cell_features['actin_branching_points'] = np.sum(ndimage.generic_filter(actin_skeleton, np.sum, size=3) > 3)
        
        features.append(cell_features)
    
    return pd.DataFrame(features)

def analyze_organelle_interactions(features):
    """Analyze interactions between organelles."""
    nucleus_centroids = np.array(features[['nucleus_centroid']].tolist())
    mito_centroids = np.array(features[['mito_centroid']].tolist())
    golgi_centroids = np.array(features[['golgi_centroid']].tolist())
    
    nuc_mito_dist = cdist(nucleus_centroids, mito_centroids)
    nuc_golgi_dist = cdist(nucleus_centroids, golgi_centroids)
    mito_golgi_dist = cdist(mito_centroids, golgi_centroids)
    
    features['avg_nuc_mito_dist'] = np.mean(nuc_mito_dist, axis=1)
    features['avg_nuc_golgi_dist'] = np.mean(nuc_golgi_dist, axis=1)
    features['avg_mito_golgi_dist'] = np.mean(mito_golgi_dist, axis=1)
    
    return features

def perform_pca(features):
    """Perform PCA on the feature set."""
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(scaled_features)
    
    return pd.DataFrame(data=pca_result, columns=['PC1', 'PC2'])

def cluster_compounds(pca_result, n_clusters=5):
    """Cluster compounds based on PCA results."""
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(pca_result)
    return clusters

def visualize_results(pca_result, clusters, condition):
    """Visualize PCA and clustering results."""
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(pca_result['PC1'], pca_result['PC2'], c=clusters, cmap='viridis')
    plt.colorbar(scatter)
    plt.title(f'PCA and Clustering of {condition} Cells')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.savefig(f'{condition}_pca_clusters.png')
    plt.close()

def compare_conditions(ko_features, inhibitor_features, control_features):
    """Compare features across different conditions."""
    comparison_results = []
    for feature in ko_features.columns:
        if feature not in ['nucleus_centroid', 'mito_centroid', 'golgi_centroid']:
            f_stat, p_value = stats.f_oneway(ko_features[feature], 
                                             inhibitor_features[feature], 
                                             control_features[feature])
            comparison_results.append({
                'feature': feature,
                'f_statistic': f_stat,
                'p_value': p_value
            })
    return pd.DataFrame(comparison_results)

def main(ko_paths, inhibitor_paths, control_paths):
    conditions = {
        'EGFR_KO': ko_paths,
        'EGFR_Inhibitor': inhibitor_paths,
        'Control': control_paths
    }
    
    all_features = {}
    all_pca_results = {}
    all_clusters = {}
    
    for condition, paths in conditions.items():
        condition_features = []
        for image_paths in paths:
            channels = load_and_preprocess_images(image_paths)
            mask = segment_composite(channels)
            features = extract_features(channels, mask)
            features = analyze_organelle_interactions(features)
            condition_features.append(features)
        
        all_features[condition] = pd.concat(condition_features, ignore_index=True)
        all_pca_results[condition] = perform_pca(all_features[condition])
        all_clusters[condition] = cluster_compounds(all_pca_results[condition])
        
        visualize_results(all_pca_results[condition], all_clusters[condition], condition)
    
    # Compare 
    comparison_results = compare_conditions(all_features['EGFR_KO'], 
                                            all_features['EGFR_Inhibitor'], 
                                            all_features['Control'])
    
    for condition, features in all_features.items():
        features.to_csv(f'{condition}_features.csv', index=False)
    comparison_results.to_csv('condition_comparison.csv', index=False)
    
    print("Analysis complete. Results saved to CSV files.")
    return all_features, all_pca_results, all_clusters, comparison_results

# usage
ko_paths = [['path/to/ko_channel1.tif', 'path/to/ko_channel2.tif', ...] for _ in range(n_ko_images)]
inhibitor_paths = [['path/to/inhibitor_channel1.tif', 'path/to/inhibitor_channel2.tif', ...] for _ in range(n_inhibitor_images)]
control_paths = [['path/to/control_channel1.tif', 'path/to/control_channel2.tif', ...] for _ in range(n_control_images)]

results = main(ko_paths, inhibitor_paths, control_paths)


# [Previous functions remain the same: load_image, normalize_intensities, segment_image, etc.]

def extract_advanced_features(image, mask, channel_type):
    """Extract advanced features from an image using a mask, considering channel type."""
    labeled_mask = measure.label(mask)
    props = measure.regionprops(labeled_mask, intensity_image=image)
    
    features = {
        'area': [], 'perimeter': [], 'eccentricity': [], 'solidity': [],
        'mean_intensity': [], 'max_intensity': [], 'min_intensity': [],
        'euler_number': [], 'extent': [], 'orientation': []
    }
    
    for region in props:
        for key in features.keys():
            features[key].append(getattr(region, key))
        
        # Add texture features
        glcm = feature.graycomatrix(region.intensity_image.astype(np.uint8), 
                                    distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], 
                                    symmetric=True, normed=True)
        features['contrast'] = feature.graycoprops(glcm, 'contrast')[0, 0]
        features['dissimilarity'] = feature.graycoprops(glcm, 'dissimilarity')[0, 0]
        features['homogeneity'] = feature.graycoprops(glcm, 'homogeneity')[0, 0]
        features['energy'] = feature.graycoprops(glcm, 'energy')[0, 0]
        features['correlation'] = feature.graycoprops(glcm, 'correlation')[0, 0]
    
    if channel_type == 'nucleus':
        features['chromatin_condensation'] = [np.std(region.intensity_image) for region in props]
    elif channel_type == 'mitochondria':
        features['mito_fragmentation'] = [len(measure.label(region.image)) for region in props]
    elif channel_type == 'golgi':
        features['golgi_dispersion'] = [np.mean(ndimage.distance_transform_edt(region.image)) for region in props]
    
    return features

def analyze_morphological_changes(features_df):
    """Analyze morphological changes across conditions."""
    conditions = features_df['condition'].unique()
    morphological_changes = {}
    
    for feature in features_df.columns:
        if feature not in ['condition', 'image_name']:
            f_stat, p_value = stats.f_oneway(*[group[feature].values for name, group in features_df.groupby('condition')])
            if p_value < 0.05:
                tukey = pairwise_tukeyhsd(features_df[feature], features_df['condition'])
                significant_pairs = [(pair[0], pair[1]) for pair in tukey.summary().data[1:] if pair[6] < 0.05]
                if significant_pairs:
                    morphological_changes[feature] = significant_pairs
    
    return morphological_changes

def visualize_morphological_changes(features_df, morphological_changes):
    """Visualize significant morphological changes."""
    for feature, pairs in morphological_changes.items():
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='condition', y=feature, data=features_df)
        plt.title(f'Distribution of {feature} Across Conditions')
        plt.savefig(f'{feature}_distribution.png')
        plt.close()

def perform_tsne(features):
    """Perform t-SNE for visualization."""
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    
    tsne = TSNE(n_components=2, random_state=42)
    tsne_result = tsne.fit_transform(scaled_features)
    
    return pd.DataFrame(data=tsne_result, columns=['TSNE1', 'TSNE2'])

def build_feature_correlation_network(features_df):
    """Build a network of correlated features."""
    corr_matrix = features_df.corr()
    G = nx.Graph()
    
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            if abs(corr_matrix.iloc[i, j]) > 0.7:  # Correlation threshold
                G.add_edge(corr_matrix.columns[i], corr_matrix.columns[j], weight=abs(corr_matrix.iloc[i, j]))
    
    return G

def visualize_feature_network(G):
    """Visualize the feature correlation network."""
    plt.figure(figsize=(12, 12))
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='lightblue', 
            node_size=500, font_size=8, font_weight='bold')
    edge_weights = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_weights)
    plt.title('Feature Correlation Network')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('feature_correlation_network.png')
    plt.close()





def main():
    # [Previous main function content]
    
    # Additional analysis steps
    morphological_changes = analyze_morphological_changes(combined_results)
    visualize_morphological_changes(combined_results, morphological_changes)
    
    tsne_results = perform_tsne(combined_results.drop(['condition', 'image_name'], axis=1))
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x='TSNE1', y='TSNE2', hue='condition', data=pd.concat([tsne_results, combined_results['condition']], axis=1))
    plt.title('t-SNE Visualization of Cell Features')
    plt.savefig('tsne_visualization.png')
    plt.close()
    
    feature_network = build_feature_correlation_network(combined_results.drop(['condition', 'image_name'], axis=1))
    visualize_feature_network(feature_network)
    
  
    
    with open('analysis_results.txt', 'w') as f:
        f.write("Significant Morphological Changes:\n")
        for feature, pairs in morphological_changes.items():
            f.write(f"{feature}: {pairs}\n")
        
        f.write("\nTemporal Changes:\n")
        for feature, change in temporal_changes.items():
            f.write(f"{feature}: Slope = {change['slope']}, p-value = {change['p_value']}\n")
    
    print("Advanced analysis complete. Additional results and visualizations have been saved.")

if __name__ == '__main__':
    main()






    #additons

import numpy as np
from scipy.stats import pearsonr

def calculate_overlap(mask1, mask2):
    """Calculate the percentage overlap between two binary masks."""
    overlap = mask1 & mask2
    return np.sum(overlap) / np.sum(mask1) * 100 if np.sum(mask1) > 0 else 0

def manders_overlap_coefficient(mask1, mask2):
    """Compute Manders' Overlap Coefficient between two binary masks."""
    overlap = mask1 & mask2
    overlap_area = np.sum(overlap)
    return overlap_area / min(np.sum(mask1), np.sum(mask2)) if min(np.sum(mask1), np.sum(mask2)) > 0 else 0

def pearson_correlation_coefficient(mask1, mask2):
    """Compute Pearson's Correlation Coefficient between two binary masks."""
    if np.sum(mask1) == 0 or np.sum(mask2) == 0:
        return np.nan
    try:
        pearson_coeff, _ = pearsonr(mask1.flatten(), mask2.flatten())
        return pearson_coeff
    except:
        return np.nan

def extract_additional_features(channels, mask):
    """Extract additional features from segmented multi-channel image."""
    labeled_mask = measure.label(mask)
    props = measure.regionprops(labeled_mask)
    
    features = []
    for region in props:
        cell_features = {
            'cell_label': region.label,
            'cell_area': region.area,
            'cell_perimeter': region.perimeter,
            'cell_eccentricity': region.eccentricity,
            'cell_mean_intensity': region.mean_intensity,
            'cell_texture_entropy': shannon_entropy(region.intensity_image),
        }
        
        #  features extraction for nucleus, mitochondria, and golgi
        nucleus_mask = channels[0] > filters.threshold_otsu(channels[0])
        mito_mask = channels[1] > filters.threshold_otsu(channels[1])
        golgi_mask = channels[2] > filters.threshold_otsu(channels[2])
        
        cell_features.update({
            'nucleus_area': np.sum(nucleus_mask),
            'nucleus_perimeter': measure.perimeter(nucleus_mask),
            'nucleus_eccentricity': measure.regionprops(nucleus_mask.astype(int))[0].eccentricity,
            'nucleus_mean_intensity': np.mean(channels[0][nucleus_mask]),
            'mitochondria_area': np.sum(mito_mask),
            'mitochondria_perimeter': measure.perimeter(mito_mask),
            'mitochondria_eccentricity': measure.regionprops(mito_mask.astype(int))[0].eccentricity,
            'mitochondria_mean_intensity': np.mean(channels[1][mito_mask]),
            'golgi_area': np.sum(golgi_mask),
            'golgi_perimeter': measure.perimeter(golgi_mask),
            'golgi_eccentricity': measure.regionprops(golgi_mask.astype(int))[0].eccentricity,
            'golgi_mean_intensity': np.mean(channels[2][golgi_mask]),
            'mito_distance_to_nucleus': ndimage.distance_transform_edt(~nucleus_mask)[mito_mask].min(),
            'golgi_distance_to_nucleus': ndimage.distance_transform_edt(~nucleus_mask)[golgi_mask].min(),
            'mito_distance_to_golgi': ndimage.distance_transform_edt(~golgi_mask)[mito_mask].min(),
            'mito_golgi_overlap': calculate_overlap(mito_mask, golgi_mask),
            'mito_nucleus_overlap': calculate_overlap(mito_mask, nucleus_mask),
            'golgi_nucleus_overlap': calculate_overlap(golgi_mask, nucleus_mask),
            'manders_mito_golgi': manders_overlap_coefficient(mito_mask, golgi_mask),
            'pearson_mito_golgi': pearson_correlation_coefficient(channels[1][mito_mask], channels[2][golgi_mask]),
            'distance_mito_membrane': ndimage.distance_transform_edt(region.image)[mito_mask].min(),
            'distance_golgi_membrane': ndimage.distance_transform_edt(region.image)[golgi_mask].min(),
            'distance_nucleus_membrane': ndimage.distance_transform_edt(region.image)[nucleus_mask].min(),
        })
        
        features.append(cell_features)
    
    return pd.DataFrame(features)

def main(ko_paths, inhibitor_paths, control_paths):
    # ..(previous code remains  same)..
    
    for condition, paths in conditions.items():
        condition_features = []
        for image_paths in paths:
            channels = load_and_preprocess_images(image_paths)
            mask = segment_composite(channels)
            features = extract_additional_features(channels, mask)
            features = analyze_organelle_interactions(features)
            condition_features.append(features)
        
        # ..(rest of the function remains the same)..

if __name__ == '__main__':
    main(ko_paths, inhibitor_paths, control_paths)
