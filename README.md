# Privacy-Preserving Brain Tumor Detection Using Radiomics

A brain MRI analysis project exploring radiomics-based tumor classification, convolutional neural networks (CNNs), fuzzy clustering, and federated learning.

## Project Overview

This academic project explores machine learning approaches for brain tumor classification from MRI images. It includes radiomics feature extraction, dimensionality reduction using Principal Component Analysis (PCA), and classification using Random Forest models.

The project also contains CNN training code, fuzzy clustering components, and a federated learning pipeline for experimenting with privacy-preserving model training.

## Key Components

- **Radiomics:** Extract quantitative features from MRI images.
- **PCA:** Reduce the dimensionality of extracted features.
- **Random Forest:** Classify samples using radiomics features.
- **CNN:** Explore image-based classification.
- **Fuzzy clustering:** Investigate clustering-based image analysis.
- **Federated learning:** Experiment with distributed model training.
- **Web interface:** Frontend and backend application code.

## Experimental Results

The recorded experiments used 5,600 training images and 1,600 testing images across four classes: glioma, meningioma, no tumor, and pituitary.

| Model | Method | Recorded Accuracy |
|---|---|---:|
| CNN Only | Deep CNN features | 76.94% |
| Radiomics Only | Random Forest + PCA | 86.75% |
| CNN + Radiomics | Combined features | 82.69% |
| Federated Radiomics | Fuzzy C-Means + PCA + FedAvg | 74.00% |

The federated experiment used four clients and four aggregation rounds. Recorded global accuracy increased from 65.31% in Round 1 to 74.00% in Round 4. PCA reduced 93 radiomics features to 12 components, accounting for 95.77% of the reported explained variance.

These figures are transcribed from the project's experimental results and have not been independently reproduced during repository preparation.

## Technologies

Python, machine learning, radiomics, CNNs, PCA, Random Forest, fuzzy clustering, federated learning, and web development.

## Repository Structure

- `backend/` — Backend application and model evaluation scripts.
- `frontend/` — User interface.
- `federated_learning/` — Federated training and evaluation scripts.
- `fuzzy_clustering/` — Fuzzy clustering implementation.
- `preprocessing/` — Image preprocessing code.
- `radiomics/` — Radiomics feature extraction.
- `FINAL_RESULTS.txt` — Detailed recorded experimental results.
- `results_summary.csv` — Results summary.
- `train_model.py` — Model training script.

## Setup

1. Clone this repository.
2. Create and activate a Python virtual environment.
3. Install the dependencies required by the relevant scripts.
4. Obtain the dataset and trained model artifacts separately.
5. Review the scripts for any local file paths before running them.

Exact installation commands and model download instructions remain to be verified.

## Important Notes

- The raw MRI dataset and trained model artifacts are excluded from this repository.
- The application may require additional model files and feature data to run.
- This is an academic research project and is **not intended for clinical diagnosis or treatment decisions**.
- Confirm dataset licensing and applicable privacy requirements before sharing any data.

## Author

Computer Science and Engineering (Data Science) student.
