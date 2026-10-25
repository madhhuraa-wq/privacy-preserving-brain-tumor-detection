# Privacy-Preserving Brain Tumor Detection Using Radiomics

A brain MRI analysis project exploring radiomics-based tumor classification, fuzzy clustering, convolutional neural networks (CNNs), and federated learning.

## Project Overview

This project explores machine learning techniques for brain tumor classification from MRI images. It includes radiomics feature extraction, dimensionality reduction using Principal Component Analysis (PCA), and classification using Random Forest models.

The project also contains CNN training code, fuzzy clustering components, and a federated learning pipeline for experimentation with privacy-preserving model training.

## Key Components

- **Radiomics:** Extract quantitative features from MRI images.
- **PCA:** Reduce the dimensionality of extracted features.
- **Random Forest:** Classify samples using radiomics features.
- **CNN:** Explore image-based classification.
- **Fuzzy clustering:** Investigate clustering-based image analysis.
- **Federated learning:** Experiment with distributed model training.
- **Web interface:** Frontend and backend application code for the project.

## Repository Structure

```text
BrainTumorProject/
├── backend/
├── frontend/
├── dataset/              # Excluded from Git
├── features/
├── federated_learning/
├── fuzzy_clustering/
├── models/               # Trained artifacts excluded from Git
├── preprocessing/
├── radiomics/
├── train_model.py
├── results_summary.csv
├── FINAL_RESULTS.txt
└── .gitignore
```

## Technologies

Python, machine learning, radiomics, CNNs, PCA, Random Forest, fuzzy clustering, federated learning, and web development.

## Results

See `FINAL_RESULTS.txt` and `results_summary.csv` for the recorded experimental results. Results may vary by model, dataset split, and evaluation method. Report each metric together with its corresponding pipeline and evaluation split.

## Setup

1. Clone this repository.
2. Create and activate a Python virtual environment.
3. Install the dependencies required by the relevant scripts.
4. Obtain the required dataset and trained model artifacts separately.
5. Update any local file paths if needed.
6. Run the appropriate training script or application entry point.

Exact installation commands and model download instructions will be added after the dependencies and application paths are verified.

## Important Notes

- The raw MRI dataset and trained model artifacts are excluded from this repository.
- The web application requires its corresponding model artifacts and testing features to run.
- This is an academic research project and is not intended for clinical diagnosis or treatment decisions.
- Confirm dataset licensing and applicable privacy requirements before sharing any data.

## Author

Computer Science and Engineering (Data Science) student.

## Experimental Results

The recorded experiments used 5,600 training images and 1,600 testing images across four classes: glioma, meningioma, no tumor, and pituitary.

| Model | Method | Recorded Test Accuracy |
|---|---|---:|
| CNN Only | Deep CNN features | 76.94% |
| Radiomics Only | Random Forest + PCA | 86.75% |
| CNN + Radiomics | Random Forest on combined features | 82.69% |
| Federated Radiomics | Fuzzy C-Means + PCA + FedAvg | 74.00% |

The federated experiment used four clients and four aggregation rounds. Its recorded global test accuracy increased from 65.31% in Round 1 to 74.00% in Round 4. PCA reduced the original 93 radiomics features to 12 components, accounting for 95.77% of the reported explained variance.

These values are transcribed from the project's experimental results files and have not been independently reproduced during repository preparation.
