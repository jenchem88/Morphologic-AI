#This Cell Morphology Analyzer extracts key morphological features from cell images, 
# which can help identify changes in cell shape and structure due to pathogen infection or drug treatment.


import numpy as np
import pandas as pd
from skimage import io, measure, morphology
from scipy import ndimage
import matplotlib.pyplot as plt

def load_and_preprocess_image(image_path):
    image = io.imread(image_path)
    # Assuming the image is grayscale. If not, convert to grayscale here.
    return image

def segment_cells(image):
    # Simple thresholding for segmentation. Adjust as needed.
    thresh = threshold_otsu(image)
    binary = image > thresh
    # Remove small objects
    cleaned = morphology.remove_small_objects(binary, min_size=50)
    # Label connected components
    labeled = measure.label(cleaned)
    return labeled

def extract_morphological_features(labeled_image):
    props = measure.regionprops(labeled_image)
    features = []
    for prop in props:
        features.append({
            'area': prop.area,
            'perimeter': prop.perimeter,
            'eccentricity': prop.eccentricity,
            'solidity': prop.solidity,
            'extent': prop.extent,
            'orientation': prop.orientation
        })
    return pd.DataFrame(features)

def analyze_morphology(image_path):
    image = load_and_preprocess_image(image_path)
    labeled = segment_cells(image)
    features = extract_morphological_features(labeled)
    
    # Basic statistical analysis
    summary = features.describe()
    
    # Visualize distributions
    features.hist(figsize=(12, 8))
    plt.tight_layout()
    plt.savefig('morphology_distributions.png')
    
    return summary, features

if __name__ == "__main__":
    image_path = "path/to/your/image.tif"
    summary, features = analyze_morphology(image_path)
    print(summary)
    features.to_csv('morphological_features.csv', index=False)
