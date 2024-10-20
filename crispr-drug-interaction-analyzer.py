import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from statsmodels.stats.multitest import multipletests
import argparse
import logging

def load_data(drug_file, crispr_file):
    drug_data = pd.read_csv(drug_file)
    crispr_data = pd.read_csv(crispr_file)
    return drug_data, crispr_data

def preprocess_data(drug_data, crispr_data):
    # Merge drug and CRISPR data
    merged_data = pd.merge(drug_data, crispr_data, on='cell_id')
    
    # Separate features and metadata
    features = merged_data.drop(['cell_id', 'drug_name', 'concentration', 'gene'], axis=1)
    metadata = merged_data[['cell_id', 'drug_name', 'concentration', 'gene']]
    
    # Normalize features
    scaler = StandardScaler()
    normalized_features = scaler.fit_transform(features)
    
    return normalized_features, metadata, features.columns

def perform_pca(data, n_components=50):
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(data)
    return pca_result, pca

def analyze_gene_drug_interactions(pca_result, metadata):
    interactions = []
    
    for drug in metadata['drug_name'].unique():
        for gene in metadata['gene'].unique():
            drug_mask = (metadata['drug_name'] == drug) & (metadata['gene'] == gene)
            control_mask = (metadata['drug_name'] == 'DMSO') & (metadata['gene'] == 'non-targeting')
            
            if drug_mask.sum() > 0 and control_mask.sum() > 0:
                drug_response = pca_result[drug_mask]
                control_response = pca_result[control_mask]
                
                t_stat, p_value = stats.ttest_ind(drug_response, control_response)
                
                effect_size = np.mean(drug_response, axis=0) - np.mean(control_response, axis=0)
                
                interactions.append({
                    'drug': drug,
                    'gene': gene,
                    'p_value': p_value[0],
                    'effect_size': effect_size[0]
                })
    
    interactions_df = pd.DataFrame(interactions)
    interactions_df['adjusted_p_value'] = multipletests(interactions_df['p_value'], method='fdr_bh')[1]
    return interactions_df

def visualize_interactions(interactions_df):
    plt.figure(figsize=(12, 8))
    sns.scatterplot(data=interactions_df, x='effect_size', y='-log10(adjusted_p_value)', 
                    hue='drug', size='-log10(adjusted_p_value)', sizes=(20, 200), alpha=0.7)
    plt.title('Gene-Drug Interactions')
    plt.xlabel('Effect Size')
    plt.ylabel('-log10(Adjusted p-value)')
    plt.axhline(y=-np.log10(0.05), color='r', linestyle='--')
    plt.axvline(x=0, color='r', linestyle='--')
    plt.savefig('gene_drug_interactions.png')
    plt.close()

def generate_top_interactions_report(interactions_df, top_n=20):
    top_interactions = interactions_df.sort_values('adjusted_p_value').head(top_n)
    
    with open('top_gene_drug_interactions.txt', 'w') as f:
        f.write("Top Gene-Drug Interactions\n")
        f.write("==========================\n\n")
        for _, row in top_interactions.iterrows():
            f.write(f"Drug: {row['drug']}\n")
            f.write(f"Gene: {row['gene']}\n")
            f.write(f"Adjusted p-value: {row['adjusted_p_value']:.6f}\n")
            f.write(f"Effect size: {row['effect_size']:.4f}\n")
            f.write("\n")

def main(drug_file, crispr_file):
    logging.basicConfig(level=logging.INFO)
    
    # Load and preprocess data
    logging.info("Loading and preprocessing data...")
    drug_data, crispr_data = load_data(drug_file, crispr_file)
    normalized_features, metadata, feature_names = preprocess_data(drug_data, crispr_data)
    
    # Perform PCA
    logging.info("Performing PCA...")
    pca_result, pca = perform_pca(normalized_features)
    
    # Analyze gene-drug interactions
    logging.info("Analyzing gene-drug interactions...")
    interactions_df = analyze_gene_drug_interactions(pca_result, metadata)
    
    # Visualize interactions
    logging.info("Generating visualizations...")
    visualize_interactions(interactions_df)
    
    # Generate report
    logging.info("Generating report...")
    generate_top_interactions_report(interactions_df)
    
    logging.info("Analysis complete. Check output files for results.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CRISPR-Drug Interaction Analyzer")
    parser.add_argument("drug_file", help="Path to the CSV file containing drug response data")
    parser.add_argument("crispr_file", help="Path to the CSV file containing CRISPR perturbation data")
    args = parser.parse_args()
    main(args.drug_file, args.crispr_file)
