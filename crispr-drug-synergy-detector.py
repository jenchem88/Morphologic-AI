# identifies potential synergistic effects between CRISPR perturbations and drug treatments
# which could reveal new combination therapies

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(crispr_file, drug_file):
    crispr_data = pd.read_csv(crispr_file)
    drug_data = pd.read_csv(drug_file)
    return crispr_data, drug_data

def calculate_synergy_scores(crispr_data, drug_data):
    synergy_scores = []
    for gene in crispr_data['gene'].unique():
        for drug in drug_data['drug'].unique():
            crispr_effect = crispr_data[crispr_data['gene'] == gene]['effect'].mean()
            drug_effect = drug_data[drug_data['drug'] == drug]['effect'].mean()
            combined_effect = (crispr_data[crispr_data['gene'] == gene]['effect'] * 
                               drug_data[drug_data['drug'] == drug]['effect']).mean()
            
            synergy_score = combined_effect - (crispr_effect * drug_effect)
            synergy_scores.append({
                'gene': gene,
                'drug': drug,
                'synergy_score': synergy_score
            })
    
    return pd.DataFrame(synergy_scores)

def plot_synergy_heatmap(synergy_scores):
    pivot_scores = synergy_scores.pivot(index='gene', columns='drug', values='synergy_score')
    plt.figure(figsize=(12, 8))
    sns.heatmap(pivot_scores, cmap='coolwarm', center=0)
    plt.title('CRISPR-Drug Synergy Scores')
    plt.tight_layout()
    plt.savefig('crispr_drug_synergy_heatmap.png')

def detect_synergies(crispr_file, drug_file):
    crispr_data, drug_data = load_data(crispr_file, drug_file)
    synergy_scores = calculate_synergy_scores(crispr_data, drug_data)
    plot_synergy_heatmap(synergy_scores)
    return synergy_scores

if __name__ == "__main__":
    synergy_scores = detect_synergies('crispr_effects.csv', 'drug_effects.csv')
    synergy_scores.to_csv('crispr_drug_synergy_scores.csv', index=False)
