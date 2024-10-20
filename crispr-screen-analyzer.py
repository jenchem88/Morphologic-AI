#performs comprehensive analysis of CRISPR screen data,
#  including dimensionality reduction, clustering, and network analysis.
#builds a machine learning model to predict gene essentiality, 
# which could be used for prioritizing genes in future screens.
#generates a gene interaction network

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.metrics import silhouette_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import networkx as nx
from tqdm import tqdm
import argparse
import logging
import os

class CRISPRScreenAnalyzer:
    def __init__(self, data_file, output_dir):
        self.data = pd.read_csv(data_file)
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.logger = self._setup_logger()

    def _setup_logger(self):
        logger = logging.getLogger('CRISPRScreenAnalyzer')
        logger.setLevel(logging.INFO)
        fh = logging.FileHandler(os.path.join(self.output_dir, 'analysis.log'))
        fh.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        return logger

    def preprocess_data(self):
        self.logger.info("Preprocessing data...")
        # Assuming data has columns: 'gene', 'guide_sequence', 'feature1', 'feature2', ...
        feature_cols = [col for col in self.data.columns if col not in ['gene', 'guide_sequence']]
        
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(self.data[feature_cols])
        
        self.processed_data = pd.DataFrame(scaled_features, columns=feature_cols)
        self.processed_data['gene'] = self.data['gene']
        self.processed_data['guide_sequence'] = self.data['guide_sequence']

    def perform_pca(self):
        self.logger.info("Performing PCA...")
        feature_cols = [col for col in self.processed_data.columns if col not in ['gene', 'guide_sequence']]
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(self.processed_data[feature_cols])
        
        self.pca_data = pd.DataFrame(data=pca_result, columns=['PC1', 'PC2'])
        self.pca_data['gene'] = self.processed_data['gene']
        self.pca_data['guide_sequence'] = self.processed_data['guide_sequence']

        # Visualize PCA results
        plt.figure(figsize=(12, 8))
        sns.scatterplot(data=self.pca_data, x='PC1', y='PC2', hue='gene', alpha=0.7)
        plt.title('PCA of CRISPR Screen Data')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'pca_visualization.png'))
        plt.close()

    def cluster_genes(self):
        self.logger.info("Clustering genes...")
        feature_cols = [col for col in self.processed_data.columns if col not in ['gene', 'guide_sequence']]
        
        # Compute gene-level aggregates
        gene_data = self.processed_data.groupby('gene')[feature_cols].mean()
        
        # Perform hierarchical clustering
        dist_matrix = pdist(gene_data)
        linkage_matrix = linkage(dist_matrix, method='ward')
        
        # Determine optimal number of clusters using silhouette score
        silhouette_scores = []
        max_clusters = min(20, len(gene_data) - 1)
        for n_clusters in range(2, max_clusters + 1):
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(gene_data)
            silhouette_scores.append(silhouette_score(gene_data, cluster_labels))
        
        optimal_clusters = silhouette_scores.index(max(silhouette_scores)) + 2
        
        # Perform K-means clustering with optimal number of clusters
        kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
        gene_data['cluster'] = kmeans.fit_predict(gene_data)
        
        # Visualize clustering results
        plt.figure(figsize=(12, 8))
        dendrogram(linkage_matrix)
        plt.title('Hierarchical Clustering of Genes')
        plt.xlabel('Gene')
        plt.ylabel('Distance')
        plt.savefig(os.path.join(self.output_dir, 'gene_clustering_dendrogram.png'))
        plt.close()
        
        plt.figure(figsize=(12, 8))
        sns.scatterplot(data=gene_data.reset_index(), x='PC1', y='PC2', hue='cluster', style='gene', s=100)
        plt.title('Gene Clusters in PCA Space')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'gene_clusters_pca.png'))
        plt.close()
        
        return gene_data

    def analyze_guide_efficiency(self):
        self.logger.info("Analyzing guide efficiency...")
        feature_cols = [col for col in self.processed_data.columns if col not in ['gene', 'guide_sequence']]
        
        guide_effects = []
        for gene in tqdm(self.processed_data['gene'].unique(), desc="Analyzing genes"):
            gene_data = self.processed_data[self.processed_data['gene'] == gene]
            control_data = self.processed_data[self.processed_data['gene'] == 'non-targeting']
            
            for guide in gene_data['guide_sequence'].unique():
                guide_data = gene_data[gene_data['guide_sequence'] == guide]
                
                effects = {}
                for feature in feature_cols:
                    t_stat, p_value = ttest_ind(guide_data[feature], control_data[feature])
                    effect_size = (guide_data[feature].mean() - control_data[feature].mean()) / control_data[feature].std()
                    effects[feature] = effect_size
                
                guide_effects.append({
                    'gene': gene,
                    'guide_sequence': guide,
                    'mean_effect_size': np.mean(list(effects.values())),
                    'max_effect_size': np.max(list(effects.values())),
                    'features_affected': sum([abs(effect) > 1 for effect in effects.values()])
                })
        
        guide_effects = pd.DataFrame(guide_effects)
        
        # Visualize guide efficiency
        plt.figure(figsize=(12, 8))
        sns.boxplot(x='gene', y='mean_effect_size', data=guide_effects)
        plt.title('Guide Efficiency by Gene')
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'guide_efficiency_by_gene.png'))
        plt.close()
        
        return guide_effects

    def identify_essential_genes(self):
        self.logger.info("Identifying essential genes...")
        feature_cols = [col for col in self.processed_data.columns if col not in ['gene', 'guide_sequence']]
        
        gene_effects = []
        for gene in tqdm(self.processed_data['gene'].unique(), desc="Analyzing genes"):
            gene_data = self.processed_data[self.processed_data['gene'] == gene]
            control_data = self.processed_data[self.processed_data['gene'] == 'non-targeting']
            
            effects = {}
            for feature in feature_cols:
                t_stat, p_value = ttest_ind(gene_data[feature], control_data[feature])
                effect_size = (gene_data[feature].mean() - control_data[feature].mean()) / control_data[feature].std()
                effects[feature] = effect_size
            
            gene_effects.append({
                'gene': gene,
                'mean_effect_size': np.mean(list(effects.values())),
                'max_effect_size': np.max(list(effects.values())),
                'features_affected': sum([abs(effect) > 1 for effect in effects.values()])
            })
        
        gene_effects = pd.DataFrame(gene_effects)
        gene_effects['p_value'] = multipletests(gene_effects['mean_effect_size'], method='fdr_bh')[1]
        gene_effects = gene_effects.sort_values('p_value')
        
        # Visualize top essential genes
        top_genes = gene_effects.head(20)
        plt.figure(figsize=(12, 8))
        sns.barplot(x='mean_effect_size', y='gene', data=top_genes)
        plt.title('Top 20 Essential Genes')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'top_essential_genes.png'))
        plt.close()
        
        return gene_effects

    def predict_gene_essentiality(self, gene_effects):
        self.logger.info("Building gene essentiality prediction model...")
        feature_cols = [col for col in self.processed_data.columns if col not in ['gene', 'guide_sequence']]
        
        # Prepare data for machine learning
        X = self.processed_data.groupby('gene')[feature_cols].mean()
        y = (gene_effects['p_value'] < 0.05).astype(int)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train Random Forest classifier
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        self.logger.info(f"Model performance: Accuracy={accuracy:.2f}, Precision={precision:.2f}, Recall={recall:.2f}, F1={f1:.2f}")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': clf.feature_importances_
        }).sort_values('importance', ascending=False)
        
        plt.figure(figsize=(12, 8))
        sns.barplot(x='importance', y='feature', data=feature_importance.head(20))
        plt.title('Top 20 Features for Predicting Gene Essentiality')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'gene_essentiality_feature_importance.png'))
        plt.close()
        
        return clf, feature_importance

    def generate_gene_network(self, gene_effects):
        self.logger.info("Generating gene interaction network...")
        feature_cols = [col for col in self.processed_data.columns if col not in ['gene', 'guide_sequence']]
        gene_data = self.processed_data.groupby('gene')[feature_cols].mean()
        
        # Compute pairwise correlations
        correlations = gene_data.T.corr()
        
        # Create network
        G = nx.Graph()
        for gene1 in correlations.index:
            for gene2 in correlations.columns:
                if gene1 != gene2:
                    correlation = correlations.loc[gene1, gene2]
                    if abs(correlation) > 0.7:  # Only keep strong correlations
                        G.add_edge(gene1, gene2, weight=abs(correlation))
        
        # Visualize network
        plt.figure(figsize=(16, 16))
        pos = nx.spring_layout(G)
        nx.draw_networkx_nodes(G, pos, node_size=100, node_color='lightblue')
        nx.draw_networkx_edges(G, pos, alpha=0.3)
        nx.draw_networkx_labels(G, pos, font_size=8)
        plt.title('Gene Interaction Network')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'gene_interaction_network.png'))
        plt.close()
        
        return G

    def run_analysis(self):
        self.preprocess_data()
        self.perform_pca()
        gene_clusters = self.cluster_genes()
        guide_efficiency = self.analyze_guide_efficiency()
        gene_effects = self.identify_essential_genes()
        essentiality_model, feature_importance = self.predict_gene_essentiality(gene_effects)
        gene_network = self.generate_gene_network(gene_effects)
        
        self.logger.info("Analysis completed successfully.")

def main(data_file, output_dir):
    analyzer = CRISPRScreenAnalyzer(data_file, output_dir)
    analyzer.run_analysis()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CRISPR Screen Analyzer")
    parser.add_argument("data_file", help="CSV file containing CRISPR screen data")
    parser.add_argument("output_dir", help="Directory to save output files")
    args = parser.parse_args()
    
    main(args.data_file, args.output_dir)
