import pandas as pd
from sklearn.decomposition import PCA
import joblib
import os

# ============================================================
# PCA FOR RADIOMICS FEATURES
# ============================================================

INPUT_FILE = "radiomics/results/radiomics_features_scaled.csv"
OUTPUT_FILE = "radiomics/results/radiomics_features_pca.csv"
PCA_FILE = "models/radiomics_pca.pkl"

print("=" * 60)
print("RADIOMICS PCA")
print("=" * 60)

# ------------------------------------------------------------
# Check input
# ------------------------------------------------------------

if not os.path.exists(INPUT_FILE):
    print("ERROR: Input file not found!")
    print(os.path.abspath(INPUT_FILE))
    exit()

# ------------------------------------------------------------
# Load scaled radiomics
# ------------------------------------------------------------

df = pd.read_csv(INPUT_FILE)

print("\nInput shape:", df.shape)

# ------------------------------------------------------------
# Separate metadata
# ------------------------------------------------------------

metadata_columns = ["image", "class"]

feature_columns = [
    col for col in df.columns
    if col not in metadata_columns
]

X = df[feature_columns]

print("Radiomics features:", X.shape[1])
print("Samples:", X.shape[0])

# ------------------------------------------------------------
# PCA
# ------------------------------------------------------------

print("\nApplying PCA...")

pca = PCA(n_components=0.95)

X_pca = pca.fit_transform(X)

print("Original features:", X.shape[1])
print("PCA components:", X_pca.shape[1])

print(
    "Explained variance:",
    pca.explained_variance_ratio_.sum()
)

# ------------------------------------------------------------
# Create PCA dataframe
# ------------------------------------------------------------

pca_columns = [
    f"PC{i+1}"
    for i in range(X_pca.shape[1])
]

pca_df = pd.DataFrame(
    X_pca,
    columns=pca_columns
)

pca_df.insert(
    0,
    "class",
    df["class"].values
)

pca_df.insert(
    0,
    "image",
    df["image"].values
)

# ------------------------------------------------------------
# Save PCA features
# ------------------------------------------------------------

os.makedirs(
    os.path.dirname(OUTPUT_FILE),
    exist_ok=True
)

pca_df.to_csv(
    OUTPUT_FILE,
    index=False
)

# ------------------------------------------------------------
# SAVE PCA MODEL
# ------------------------------------------------------------

os.makedirs(
    os.path.dirname(PCA_FILE),
    exist_ok=True
)

joblib.dump(
    pca,
    PCA_FILE
)

# ------------------------------------------------------------
# Final information
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("PCA COMPLETED SUCCESSFULLY")
print("=" * 60)

print("\nPCA feature file:")
print(os.path.abspath(OUTPUT_FILE))

print("\nPCA model:")
print(os.path.abspath(PCA_FILE))

print("\nFinal shape:", pca_df.shape)

print("\nFirst 5 rows:")
print(pca_df.head())