# Morphologic-AI

Morphologic AI
## Introduction

https://www.youtube.com/watch?v=A6Y1BBzHO-U

Welcome to the Morphologic AI project, where we're bridging the gap between high-content cellular imaging and interpretable machine learning for accelerated drug discovery. Our diverse, international team brings together expertise from various fields to tackle the complex challenges of decoding cellular responses to drug treatments and genetic perturbations.

## Meet Our Team

- **Trevor McKee (PhD)** (tdmckee)
  - Role: Project Coordinator and Data Analysis Lead 
  - Expertise: Data interpretation, experimental design, and cross-functional team coordination

- **Krishiv Potluri** (kishpish)
  - Role: Machine Learning, Algorithm Development and Integration Specialist
  - Expertise: Biological feature extraction, developing and implementing various algorithms for HUVEC analysis

- **Jennifer Madrigal** (jenchem88)
  - Role: Bioinformatics and Data Integration Specialist
  - Expertise: Image data pre-processing, feature classification, and biological feature extraction

- **Arudhir Singh** ()
  - Role: Machine Learning and Cloud Infrastructure Lead
  - Expertise: AWS infrastructure, embedding analysis, benchmarking, and biological feature extraction

- **Neil Blake** 
  - Role: Image Processing and Segmentation Expert
  - Expertise: Cell segmentation algorithms, high-performance computing, and data pipeline optimization

- **Leonardo Juárez Zucco** (Lordleojz)
  - Role: Bioinformatics and Biological Data processing Specialist
  - Expertise: Bioinformatics, Massive Data processing and Biological Data analysis


_Project Scope_

Morphologic AI is an innovative project that leverages spatial information embedded in cellular images to gain insights into cell health and functionality in response to various challenges. Our focus is on analyzing Human Umbilical Vein Endothelial Cells (HUVECs) and their morphological changes in response to infectious agents and drug treatments. By utilizing fluorescent markers (Hoechst DNA, ConA, Phalloidin, Syto14, and WGA) and advanced computer vision-based segmentation approaches, we aim to measure changes in cellular morphology to identify drug-infectious disease interactions in the RxRx19a and b datasets. Our goal is to develop an automated, scalable solution for analyzing high-throughput microscopic images of HUVEC cells infected by pathogens and impacted by whole-genome CRISPR screens. This approach could potentially identify drugs that restore "normal" HUVEC morphology, presenting candidates for drug repurposing in infectious disease treatment.

_Impact_

Advancement in drug repurposing: By identifying drug-cell interactions that perturb cellular morphology, we can potentially repurpose existing drugs for treating infectious diseases, accelerating drug discovery processes.
Enhanced understanding of infectious diseases: Our workflow provides insights into how pathogens affect cellular structures, contributing to a deeper understanding of infection disease mechanisms.
Improved high-throughput screening: The development of an automated, scalable solution for analyzing microscopic images significantly enhances the efficiency of drug screening processes.
Vascular biology contributions: Focusing on HUVECs, our project contributes valuable data to the field, impacting research on vascular-related issues.

_Background & Relevance_

Morphologic AI is set to transform drug discovery by leveraging advanced computer vision technology to better understand cellular morphology changes due to disease, drug treatment, or transcriptional changes. While companies like Recursion have demonstrated the use of ML embeddings from fluorescent cell images to predict multiple cellular features, these black-box deep learning embeddings are not easily interpretable, limiting their usefulness for the broader biotech field.

_Our project aims to close this gap by_

Segmenting multi-cell images to extract individual cells and map key morphological features on a per-cell basis.
Providing a more biologically interpretable way to describe drug treatment or disease-related perturbations.
Comparing our mapped features to Recursion's embeddings to assess how well human-explainable features align with unsupervised machine learning.
Exploring whether combining interpretable features with unsupervised embeddings enhances data interpretation.

We are utilizing information from the morphological effects of 1500+ small molecules from Recursion datasets to extract high-dimensional information on drug clustering. This approach provides drug engineering teams with additional insights into the form-function relationship of current drugs, expanding beyond the limitations of low-throughput assays and SMILES correlations.
Our long-term vision includes expanding the Cell Paint concept to predict finer details of protein complexes within entire HUVEC cells by training on electron microscopy cellular images annotated via Correlative Light Electron Microscopy (CLEM). This could potentially allow us to predict the location of features as small as 200 Angstroms from fluorescently labeled images.

_Methodology_

**Data

Source: We primarily use the RxRx19a and b datasets as well as RxRx3 provided by Recursion Pharmaceuticals.
Type: High-throughput microscopic images of HUVEC cells with various fluorescent markers (Hoechst DNA, ConA, Phalloidin, Syto14, and WGA).
Preprocessing: Due to mislabeling issues in the original dataset, we pivoted to focus on EGFR-specific drug effects on HUVEC cells.

_Pipeline/Tools_

Image Segmentation: We employ advanced computer vision techniques to segment multi-cell images and extract individual cells.
Feature Extraction: We've engineered a process to extract specific biological features that provide insights into the underlying state of the cells.
Feature Mapping: Our innovative approach maps the extracted features against Recursion's embeddings to compare explainable features with unsupervised embeddings.
Drug Clustering: We utilize the morphological effects of 1500+ small molecules to perform high-dimensional clustering of drugs.

_Results & Demo_

[Note: Include screenshots, graphs, or links to interactive demos here-- maybe the video as well]
Our current results focus on the analysis of EGFR-specific drug effects on HUVEC cells. We've developed a pipeline that:

Segments individual cells from multi-cell images
Extracts and quantifies key morphological features
Maps these features against Recursion's embeddings
Visualizes drug clustering based on morphological effects

[Include a walkthrough of key features or functionality here]

_Interpretation_ 

Our results demonstrate that explainable, biologically relevant features can be extracted from cellular images and meaningfully compared to unsupervised machine learning embeddings. This approach connects black-box AI predictions and human-interpretable biological insights.
The analysis of EGFR-specific drug effects provides a deeper understanding of how these drugs impact HUVEC cell morphology, potentially revealing new insights into their mechanisms of action and off-target effects.

_Potential Impact_

If fully realized, our project could have many far-reaching implications:

Accelerated drug discovery: By providing more interpretable data on drug effects at the cellular level, we can speed up the process of identifying promising drug candidates and repurposing existing drugs.
Improved understanding of disease mechanisms: Our approach could reveal new insights into how diseases affect cell morphology, leading to new therapeutic targets.
Enhanced precision medicine: The ability to predict fine-scale cellular features could lead to more personalized treatment approaches based on individual patient cell responses.
Advancements in AI interpretability: Through the connection of AI-generated embeddings and human-interpretable features, we contribute to the broader goal of making AI more explainable and trustworthy in scientific applications.
Cross-disciplinary insights: Our methods could be applied to other cell types and biological systems, potentially leading to breakthroughs in various areas of biomedical research.

_Future Directions_

Expand analysis to include a broader range of cell types and disease models
Integrate CLEM imaging to predict finer subcellular structures
Develop more sophisticated AI models that combine interpretable features with unsupervised learning for enhanced predictive power
Collaborate with pharmaceutical companies to apply our methods in real-world drug discovery pipelines

_Contribution_

We welcome contributions from the Bio x ML scientific community. 

_Acknowledgments_

We would like to thank Recursion Pharmaceuticals for providing the datasets used in this project,ENVEDA, Polaris, ESM3, Ginkgo, Modal, AWS, RunPod, DigitalOcean as well as Lux Capital for hosting this project.
