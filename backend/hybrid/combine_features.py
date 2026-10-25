import pandas as pd
import os

# ============================================================
# COMBINE CNN + RADIOMICS FEATURES
# ============================================================

CNN_FILE = "../../features/cnn_training_features.csv"

RADIOMICS_FILE = "../../radiomics/results/radiomics_features_pca.csv"

OUTPUT_FILE = "../../features/hybrid_features.csv"

print("=" * 60)
print("COMBINING CNN + RADIOMICS FEATURES")
print("=" * 60)

# ------------------------------------------------------------
# Load CNN features
# ------------------------------------------------------------

print("\nLoading CNN features...")

cnn_df = pd.read_csv(CNN_FILE)

print("CNN shape:", cnn_df.shape)

# ------------------------------------------------------------
# Load Radiomics PCA features
# ------------------------------------------------------------

print("\nLoading radiomics PCA features...")

radiomics_df = pd.read_csv(RADIOMICS_FILE)

print("Radiomics PCA shape:", radiomics_df.shape)

# ------------------------------------------------------------
# Check image IDs
# ------------------------------------------------------------

print("\nChecking image IDs...")

print("CNN images:", cnn_df["image"].nunique())
print("Radiomics images:", radiomics_df["image"].nunique())

# ------------------------------------------------------------
# Merge using image filename
# ------------------------------------------------------------

print("\nMatching CNN and radiomics features...")

hybrid_df = pd.merge(
    cnn_df,
    radiomics_df,
    on="image",
    how="inner",
    suffixes=("_cnn", "_radiomics")
)

print("Hybrid shape after merge:", hybrid_df.shape)

# ------------------------------------------------------------
# Check class consistency
# ------------------------------------------------------------

print("\nChecking class consistency...")

if "class_cnn" in hybrid_df.columns and "class_radiomics" in hybrid_df.columns:

    mismatch = (
        hybrid_df["class_cnn"]
        != hybrid_df["class_radiomics"]
    ).sum()

    print("Class mismatches:", mismatch)

    if mismatch > 0:
        print("ERROR: Class labels do not match!")
        exit()

    hybrid_df["class"] = hybrid_df["class_cnn"]

    hybrid_df.drop(
        columns=["class_cnn", "class_radiomics"],
        inplace=True
    )

# ------------------------------------------------------------
# Save hybrid dataset
# ------------------------------------------------------------

os.makedirs(
    os.path.dirname(OUTPUT_FILE),
    exist_ok=True
)

hybrid_df.to_csv(
    OUTPUT_FILE,
    index=False
)

# ------------------------------------------------------------
# Final information
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("HYBRID FEATURE CREATION COMPLETED!")
print("=" * 60)

print("Output file:")
print(os.path.abspath(OUTPUT_FILE))

print("Final shape:", hybrid_df.shape)

print("\nClass distribution:")

print(
    hybrid_df["class"].value_counts()
)

print("\nFirst 5 rows:")

print(
    hybrid_df.iloc[:, :10].head()
)