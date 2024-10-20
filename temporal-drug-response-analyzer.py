#analyzes the temporal dynamics of drug responses, which captures how cells change over time.
#clusters drug response trajectories, potentially showing drugs with similar mechanisms of action.
#identifies temporal patterns in feature changes
#detects potentially synergistic drug pairs
#generates a drug interaction network

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests
from scipy.interpolate import interp1d
from scipy.signal import find_peaks
import networkx as nx
from tqdm import tqdm
import argparse
import logging
import os

class TemporalDrugResponseAnalyzer:
    def __init__(self, data_file, output_dir):
        self.data = pd.read_csv(data_file)
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.logger = self._setup_logger()

    def _setup_logger(self):
        logger = logging.getLogger('TemporalDrugResponseAnalyzer')
        logger.setLevel(logging.INFO)
        fh = logging.FileHandler(os.path.join(self.output_dir, 'analysis.log'))
        fh.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        return logger

    def preprocess_data(self):
        self.logger.info("Preprocessing data...")
        # Assuming data has columns: 'drug', 'concentration', 'time', 'feature1', 'feature2', ...
        feature_cols = [col for col in self.data.columns if col not in ['drug', 'concentration', 'time']]
        
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(self.data[feature_cols])
        
        self.processed_data = pd.DataFrame(scaled_features, columns=feature_cols)
        self.processed_data['drug'] = self.data['drug']
        self.processed_data['concentration'] = self.data['concentration']
        self.processed_data['time'] = self.data['time']

    def perform_pca(self):
        self.logger.info("Performing PCA...")
        feature_cols = [col for col in self.processed_data.columns if col not in ['drug', 'concentration', 'time']]
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(self.processed_data[feature_cols])
        
        self.pca_data = pd.DataFrame(data=pca_result, columns=['PC1', 'PC2'])
        self.pca_data['drug'] = self.processed_data['drug']
        self.pca_data['concentration'] = self.processed_data['concentration']
        self.pca_data['time'] = self.processed_data['time']

    def visualize_trajectories(self):
        self.logger.info("Visualizing drug response trajectories...")
        plt.figure(figsize=(12, 8))
        for drug in self.pca_data['drug'].unique():
            drug_data = self.pca_data[self.pca_data['drug'] == drug]
            plt.plot(drug_data['PC1'], drug_data['PC2'], '-o', label=drug)
        
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.title('Drug Response Trajectories')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'drug_response_trajectories.png'))
        plt.close()

    def cluster_trajectories(self):
        self.logger.info("Clustering drug response trajectories...")
        # Compute trajectory features (e.g., direction, speed, curvature)
        trajectory_features = []
        for drug in self.pca_data['drug'].unique():
            drug_data = self.pca_data[self.pca_data['drug'] == drug].sort_values('time')
            
            # Compute direction (angle between start and end points)
            start = drug_data[['PC1', 'PC2']].iloc[0]
            end = drug_data[['PC1', 'PC2']].iloc[-1]
            direction = np.arctan2(end['PC2'] - start['PC2'], end['PC1'] - start['PC1'])
            
            # Compute speed (total distance / time)
            distances = np.sqrt(np.diff(drug_data['PC1'])**2 + np.diff(drug_data['PC2'])**2)
            speed = distances.sum() / (drug_data['time'].max() - drug_data['time'].min())
            
            # Compute curvature (sum of angle changes)
            angles = np.arctan2(np.diff(drug_data['PC2']), np.diff(drug_data['PC1']))
            curvature = np.abs(np.diff(angles)).sum()
            
            trajectory_features.append({
                'drug': drug,
                'direction': direction,
                'speed': speed,
                'curvature': curvature
            })
        
        trajectory_features = pd.DataFrame(trajectory_features)
        
        # Perform clustering
        kmeans = KMeans(n_clusters=5, random_state=42)
        trajectory_features['cluster'] = kmeans.fit_predict(trajectory_features[['direction', 'speed', 'curvature']])
        
        # Visualize clusters
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(trajectory_features['direction'], trajectory_features['speed'], 
                              c=trajectory_features['cluster'], s=trajectory_features['curvature']*100, 
                              alpha=0.6, cmap='viridis')
        plt.xlabel('Direction')
        plt.ylabel('Speed')
        plt.title('Drug Trajectory Clusters')
        plt.colorbar(scatter, label='Cluster')
        for i, txt in enumerate(trajectory_features['drug']):
            plt.annotate(txt, (trajectory_features['direction'].iloc[i], trajectory_features['speed'].iloc[i]))
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'trajectory_clusters.png'))
        plt.close()
        
        return trajectory_features

    def analyze_temporal_patterns(self):
        self.logger.info("Analyzing temporal patterns...")
        feature_cols = [col for col in self.processed_data.columns if col not in ['drug', 'concentration', 'time']]
        temporal_patterns = {}
        
        for drug in tqdm(self.processed_data['drug'].unique(), desc="Analyzing drugs"):
            drug_data = self.processed_data[self.processed_data['drug'] == drug].sort_values('time')
            patterns = {}
            
            for feature in feature_cols:
                # Interpolate to get smooth curve
                interp_func = interp1d(drug_data['time'], drug_data[feature], kind='cubic')
                smooth_time = np.linspace(drug_data['time'].min(), drug_data['time'].max(), 1000)
                smooth_values = interp_func(smooth_time)
                
                # Find peaks and troughs
                peaks, _ = find_peaks(smooth_values)
                troughs, _ = find_peaks(-smooth_values)
                
                patterns[feature] = {
                    'peaks': smooth_time[peaks],
                    'troughs': smooth_time[troughs],
                    'overall_trend': np.polyfit(smooth_time, smooth_values, 1)[0]  # Linear trend
                }
            
            temporal_patterns[drug] = patterns
        
        # Visualize patterns for a sample drug and feature
        sample_drug = list(temporal_patterns.keys())[0]
        sample_feature = feature_cols[0]
        
        plt.figure(figsize=(10, 6))
        drug_data = self.processed_data[self.processed_data['drug'] == sample_drug].sort_values('time')
        plt.plot(drug_data['time'], drug_data[sample_feature], 'o-', label='Original data')
        
        interp_func = interp1d(drug_data['time'], drug_data[sample_feature], kind='cubic')
        smooth_time = np.linspace(drug_data['time'].min(), drug_data['time'].max(), 1000)
        smooth_values = interp_func(smooth_time)
        plt.plot(smooth_time, smooth_values, label='Interpolated')
        
        plt.plot(temporal_patterns[sample_drug][sample_feature]['peaks'], 
                 interp_func(temporal_patterns[sample_drug][sample_feature]['peaks']), 
                 'ro', label='Peaks')
        plt.plot(temporal_patterns[sample_drug][sample_feature]['troughs'], 
                 interp_func(temporal_patterns[sample_drug][sample_feature]['troughs']), 
                 'go', label='Troughs')
        
        plt.xlabel('Time')
        plt.ylabel(sample_feature)
        plt.title(f'Temporal Pattern for {sample_drug} - {sample_feature}')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'sample_temporal_pattern.png'))
        plt.close()
        
        return temporal_patterns

    def identify_synergistic_pairs(self):
        self.logger.info("Identifying synergistic drug pairs...")
        feature_cols = [col for col in self.processed_data.columns if col not in ['drug', 'concentration', 'time']]
        drug_effects = {}
        
        for drug in self.processed_data['drug'].unique():
            drug_data = self.processed_data[self.processed_data['drug'] == drug]
            control_data = self.processed_data[self.processed_data['drug'] == 'control']
            
            effects = {}
            for feature in feature_cols:
                t_stat, p_value = ttest_ind(drug_data[feature], control_data[feature])
                effect_size = (drug_data[feature].mean() - control_data[feature].mean()) / control_data[feature].std()
                effects[feature] = effect_size
            
            drug_effects[drug] = effects
        
        synergies = []
        drugs = list(drug_effects.keys())
        for i in range(len(drugs)):
            for j in range(i+1, len(drugs)):
                drug1, drug2 = drugs[i], drugs[j]
                synergy_score = np.mean([
                    drug_effects[drug1][feature] * drug_effects[drug2][feature]
                    for feature in feature_cols
                ])
                synergies.append({
                    'drug1': drug1,
                    'drug2': drug2,
                    'synergy_score': synergy_score
                })
        
        synergies = pd.DataFrame(synergies)
        synergies = synergies.sort_values('synergy_score', ascending=False)
        
        # Visualize top synergistic pairs
        top_synergies = synergies.head(20)
        plt.figure(figsize=(12, 8))
        sns.barplot(x='synergy_score', y='drug1', hue='drug2', data=top_synergies)
        plt.title('Top Synergistic Drug Pairs')
        plt.xlabel('Synergy Score')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'top_synergistic_pairs.png'))
        plt.close()
        
        return synergies

    def generate_network(self, synergies):
        self.logger.info("Generating drug interaction network...")
        G = nx.Graph()
        
        # Add nodes
        for drug in self.processed_data['drug'].unique():
            G.add_node(drug)
        
        # Add edges
        for _, row in synergies.iterrows():
            G.add_edge(row['drug1'], row['drug2'], weight=row['synergy_score'])
        
        # Visualize network
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(G)
        nx.draw_networkx_nodes(G, pos, node_size=300, node_color='lightblue')
        nx.draw_networkx_labels(G, pos)
        
        edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
        nx.draw_networkx_edges(G, pos, width=edge_weights, alpha=0.7, edge_color='gray')
        
        plt.title('Drug Interaction Network')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'drug_interaction_network.png'))
        plt.close()

    def run_analysis(self):
        self.preprocess_data()
        self.perform_pca()
        self.visualize_trajectories()
        trajectory_clusters = self.cluster_trajectories()
        temporal_patterns = self.analyze_temporal_patterns()
        synergies = self.identify_synergistic_pairs()
        self.generate_network(synergies)
        
        self.logger.info("Analysis completed successfully.")

def main(data_file, output_dir):
    analyzer = TemporalDrugResponseAnalyzer(data_file, output_dir)
    analyzer.run_analysis()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Temporal Drug Response Analyzer")
    parser.add_argument("data_file", help="CSV file containing temporal drug response data")
    parser.add_argument("output_dir", help="Directory to save output files")
    args = parser.parse_args()
    
    main(args.data_file, args.output_dir)
