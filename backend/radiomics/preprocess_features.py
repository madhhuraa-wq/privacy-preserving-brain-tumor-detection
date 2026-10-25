import pandas as pd
from sklearn.preprocessing import StandardScaler
import os

# ============================================================
# PREPROCESS RADIOMICS FEATURES
# ============================================================

INPUT_FILE = "../../radiomics/results/radiomics_features.csv"
OUTPUT_FILE = "../../radiomics/results/radiomics_features_scaled.csv"

print("=" * 60)
print("Loading radiomics features...")
print("=" * 60)

# ------------------------------------------------------------
# Check input file
# ------------------------------------------------------------

if not os.path.exists(INPUT_FILE):
    print("ERROR: Input file not found!")
    print("Expected:", os.path.abspath(INPUT_FILE))
    exit()

# ------------------------------------------------------------
# Load CSV
# ------------------------------------------------------------

df = pd.read_csv(INPUT_FILE)

print("Original shape:", df.shape)

# ------------------------------------------------------------
# Separate metadata from radiomics features
# ------------------------------------------------------------

metadata_columns = ["image", "class"]

feature_columns = [
    col for col in df.columns
    if col not in metadata_columns
]

print("\nMetadata columns:", metadata_columns)
print("Number of radiomics features:", len(feature_columns))

# ------------------------------------------------------------
# Check missing values
# ------------------------------------------------------------

print("\nChecking missing values...")

missing_values = df[feature_columns].isnull().sum().sum()

print("Total missing values:", missing_values)

if missing_values > 0:
    print("Filling missing values with median...")

    df[feature_columns] = df[feature_columns].fillna(
        df[feature_columns].median()
    )

# ------------------------------------------------------------
# Standardize radiomics features
# ------------------------------------------------------------

print("\nScaling radiomics features...")

scaler = StandardScaler()

df[feature_columns] = scaler.fit_transform(
    df[feature_columns]
)

# ------------------------------------------------------------
# Save processed features
# ------------------------------------------------------------

df.to_csv(OUTPUT_FILE, index=False)

print("\n" + "=" * 60)
print("Preprocessing completed successfully!")
print("=" * 60)

print("Output file:", os.path.abspath(OUTPUT_FILE))
print("Final shape:", df.shape)

print("\nFirst 5 rows:")
print(df.head())