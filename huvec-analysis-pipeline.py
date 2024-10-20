#Image Processing: It segments cells from microscopy images and extracts a wide range of morphological and texture features.
#Drug Response Analysis: It analyzes how different drugs affect cell features using PCA and visualization.
#Infection Classification: It trains a machine learning model to classify infected vs. uninfected cells.
# it quantifies the effects of CRISPR perturbations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from skimage import io, segmentation, measure, feature, filters
from scipy import ndimage
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests
import networkx as nx
import argparse
import logging
import os

class HUVECAnalysisPipeline:
    def __init__(self, image_dir, metadata_file, output_dir):
        self.image_dir = image_dir
        self.metadata = pd.read_csv(metadata_file)
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.logger = self._setup_logger()

    def _setup_logger(self):
        logger = logging.getLogger('HUVECAnalysisPipeline')
        logger.setLevel(logging.INFO)
        fh = logging.FileHandler(os.path.join(self.output_dir, 'pipeline.log'))
        fh.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        return logger

    def process_images(self):
        self.logger.info("Processing images...")
        self.cell_features = []
        for idx, row in self.metadata.iterrows():
            image_path = os.path.join(self.image_dir, row['image_filename'])
            image = io.imread(image_path)
            cell_masks = self._segment_cells(image)
            features = self._extract_features(image, cell_masks)
            features['image_id'] = row['image_id']
            features['condition'] = row['condition']
            features['crispr_target'] = row['crispr_target']
            self.cell_features.append(features)
        self.cell_features = pd.concat(self.cell_features, ignore_index=True)
        self.cell_features.to_csv(os.path.join(self.output_dir, 'cell_features.csv'), index=False)

    def _segment_cells(self, image):
        # Apply Gaussian filter to reduce noise
        smoothed = filters.gaussian(image, sigma=2)
        
        # Use Li's method for thresholding
        thresh = filters.threshold_li(smoothed)
        binary = smoothed > thresh

        # Apply watershed segmentation
        distance = ndimage.distance_transform_edt(binary)
        local_max = feature.peak_local_max(distance, indices=False, footprint=np.ones((3, 3)), labels=binary)
        markers = measure.label(local_max)
        cell_masks = segmentation.watershed(-distance, markers, mask=binary)
        
        return cell_masks

    def _extract_features(self, image, cell_masks):
        props = measure.regionprops(cell_masks, intensity_image=image)
        features = []
        for prop in props:
            cell_features = {
                'area': prop.area,
                'perimeter': prop.perimeter,
                'eccentricity': prop.eccentricity,
                'mean_intensity': prop.mean_intensity,
                'max_intensity': prop.max_intensity,
                'min_intensity': prop.min_intensity,
                'solidity': prop.solidity,
                'orientation': prop.orientation,
                'major_axis_length': prop.major_axis_length,
                'minor_axis_length': prop.minor_axis_length
            }
            # Add texture features
            cell_image = prop.intensity_image
            glcm = feature.graycomatrix(cell_image.astype(np.uint8), [1], [0, np.pi/4, np.pi/2, 3*np.pi/4], symmetric=True, normed=True)
            cell_features['contrast'] = feature.graycoprops(glcm, 'contrast')[0, 0]
            cell_features['dissimilarity'] = feature.graycoprops(glcm, 'dissimilarity')[0, 0]
            cell_features['homogeneity'] = feature.graycoprops(glcm, 'homogeneity')[0, 0]
            cell_features['energy'] = feature.graycoprops(glcm, 'energy')[0, 0]
            cell_features['correlation'] = feature.graycoprops(glcm, 'correlation')[0, 0]
            features.append(cell_features)
        return pd.DataFrame(features)

    def analyze_drug_responses(self):
        self.logger.info("Analyzing drug responses...")
        drug_responses = self.cell_features.groupby(['condition', 'crispr_target']).mean().reset_index()
        
        # Perform PCA
        feature_cols = [col for col in drug_responses.columns if col not in ['condition', 'crispr_target']]
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(drug_responses[feature_cols])
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(scaled_features)
        
        # Visualize drug responses
        plt.figure(figsize=(12, 8))
        for condition in drug_responses['condition'].unique():
            mask = drug_responses['condition'] == condition
            plt.scatter(pca_result[mask, 0], pca_result[mask, 1], label=condition)
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.title('Drug Responses in PCA Space')
        plt.legend()
        plt.savefig(os.path.join(self.output_dir, 'drug_responses_pca.png'))
        plt.close()
        
        return drug_responses

    def train_infection_classifier(self):
        self.logger.info("Training infection classifier...")
        # Assume 'infected' is a column in self.cell_features
        X = self.cell_features.drop(['image_id', 'condition', 'crispr_target', 'infected'], axis=1)
        y = self.cell_features['infected']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_train, y_train)
        
        y_pred = clf.predict(X_test)
        
        report = classification_report(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        with open(os.path.join(self.output_dir, 'infection_classifier_report.txt'), 'w') as f:
            f.write(report)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d')
        plt.title('Confusion Matrix')
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.savefig(os.path.join(self.output_dir, 'infection_classifier_confusion_matrix.png'))
        plt.close()
        
        return clf

    def analyze_crispr_effects(self):
        self.logger.info("Analyzing CRISPR effects...")
        crispr_effects = []
        for target in self.cell_features['crispr_target'].unique():
            if target == 'control':
                continue
            target_data = self.cell_features[self.cell_features['crispr_target'] == target]
            control_data = self.cell_features[self.cell_features['crispr_target'] == 'control']
            
            for feature in self.cell_features.columns:
                if feature in ['image_id', 'condition', 'crispr_target']:
                    continue
                t_stat, p_value = ttest_ind(target_data[feature], control_data[feature])
                effect_size = (target_data[feature].mean() - control_data[feature].mean()) / control_data[feature].std()
                crispr_effects.append({
                    'crispr_target': target,
                    'feature': feature,
                    'effect_size': effect_size,
                    'p_value': p_value
                })
        
        crispr_effects = pd.DataFrame(crispr_effects)
        crispr_effects['adjusted_p_value'] = multipletests(crispr_effects['p_value'], method='fdr_bh')[1]
        crispr_effects.to_csv(os.path.join(self.output_dir, 'crispr_effects.csv'), index=False)
        
        # Visualize top CRISPR effects
        top_effects = crispr_effects.sort_values('adjusted_p_value').head(20)
        plt.figure(figsize=(12, 8))
        sns.scatterplot(data=top_effects, x='effect_size', y='-log10(adjusted_p_value)', 
                        hue='crispr_target', size='-log10(adjusted_p_value)', sizes=(20, 200))
        plt.title('Top CRISPR Effects')
        plt.xlabel('Effect Size')
        plt.ylabel('-log10(Adjusted p-value)')
        plt.savefig(os.path.join(self.output_dir, 'top_crispr_effects.png'))
        plt.close()
        
        return crispr_effects

    def identify_drug_crispr_interactions(self, drug_responses, crispr_effects):
        self.logger.info("Identifying drug-CRISPR interactions...")
        interactions = []
        for drug in drug_responses['condition'].unique():
            if drug == 'control':
                continue
            drug_data = drug_responses[drug_responses['condition'] == drug]
            control_data = drug_responses[drug_responses['condition'] == 'control']
            
            for target in crispr_effects['crispr_target'].unique():
                drug_target_data = drug_data[drug_data['crispr_target'] == target]
                drug_control_data = drug_data[drug_data['crispr_target'] == 'control']
                control_target_data = control_data[control_data['crispr_target'] == target]
                control_control_data = control_data[control_data['crispr_target'] == 'control']
                
                for feature in drug_responses.columns:
                    if feature in ['condition', 'crispr_target']:
                        continue
                    
                    drug_effect = drug_control_data[feature].mean() - control_control_data[feature].mean()
                    crispr_effect = control_target_data[feature].mean() - control_control_data[feature].mean()
                    combined_effect = drug_target_data[feature].mean() - control_control_data[feature].mean()
                    
                    interaction_score = combined_effect - (drug_effect + crispr_effect)
                    
                    interactions.append({
                        'drug': drug,
                        'crispr_target': target,
                        'feature': feature,
                        'interaction_score': interaction_score
                    })
        
        interactions = pd.DataFrame(interactions)
        interactions.to_csv(os.path.join(self.output_dir, 'drug_crispr_interactions.csv'), index=False)
        
        # Visualize top interactions
        top_interactions = interactions.groupby(['drug', 'crispr_target'])['interaction_score'].mean().reset_index()
        top_interactions = top_interactions.sort_values('interaction_score', key=abs, ascending=False).head(20)
        
        plt.figure(figsize=(12, 8))
        sns.barplot(data=top_interactions, x='interaction_score', y='drug', hue='crispr_target')
        plt.title('Top Drug-CRISPR Interactions')
        plt.xlabel('Interaction Score')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'top_drug_crispr_interactions.png'))
        plt.close()
        
        return interactions

    def generate_network(self, interactions):
        self.logger.info("Generating interaction network...")
        G = nx.Graph()
        
        # Add nodes
        for drug in interactions['drug'].unique():
            G.add_node(drug, node_type='drug')
        for target in interactions['crispr_target'].unique():
            G.add_node(target, node_type='crispr_target')
        
        # Add edges
        for _, row in interactions.iterrows():
            G.add_edge(row['drug'], row['crispr_target'], weight=abs(row['interaction_score']))
        
        # Visualize network
        pos = nx.spring_layout(G)
        plt.figure(figsize=(12, 8))
        nx.draw_networkx_nodes(G, pos, node_color=['r' if G.nodes[n]['node_type'] == 'drug' else 'b' for n in G.nodes])
        nx.draw_networkx_edges(G, pos, width=[G[u][v]['weight'] for u, v in G.edges()])
        nx.draw_networkx_labels(G, pos)
        plt.title('Drug-CRISPR Interaction Network')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'drug_crispr_network.png'))
        plt.close()

    def run_pipeline(self):
        self.process_images()
        drug_responses = self.analyze_drug_responses()
        self.train_infection_classifier()
        crispr_effects = self.analyze_crispr_effects()
        interactions = self.identify_drug_crispr_interactions(drug_responses, crispr_effects)
        self.generate_network(interactions)
        self.logger.info("Pipeline completed successfully.")

def main(image_dir, metadata_file, output_dir):
    pipeline = HUVECAnalysisPipeline(image_dir, metadata_file, output_dir)
    pipeline.run_pipeline()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HUVEC Cell Analysis Pipeline")
    parser.add_argument("image_dir", help="Directory containing cell images")
    parser.add_argument("metadata_file", help="CSV file containing image metadata")
    parser.add_argument("output_dir", help="Directory to save output files")
    args = parser.parse_args()
    
    main(args.image_dir, args.metadata_file, args.output_dir)
