import pandas as pd
import os
import joblib

from sklearn.preprocessing import StandardScaler

# ============================================================
# CREATE TEST HYBRID FEATURES
# CNN + RADIOMICS PCA
# ============================================================

CNN_TEST_FILE = "features/cnn_testing_features.csv"

RAD_TRAIN_FILE = "radiomics/results/radiomics_features.csv"
RAD_TEST_FILE = "radiomics/results/radiomics_testing_features.csv"

PCA_FILE = "models/radiomics_pca.pkl"

OUTPUT_FILE = "features/hybrid_testing_features.csv"

print("=" * 60)
print("CREATING TEST HYBRID FEATURES")
print("=" * 60)

# ------------------------------------------------------------
# Check files
# ------------------------------------------------------------

print("\nChecking input files...")

files = [
    CNN_TEST_FILE,
    RAD_TRAIN_FILE,
    RAD_TEST_FILE,
    PCA_FILE
]

for file in files:
    print("\n", os.path.abspath(file))

    if os.path.exists(file):
        print("OK")
    else:
        print("ERROR: FILE NOT FOUND")
        exit()

# ------------------------------------------------------------
# Load training radiomics
# ------------------------------------------------------------

print("\nLoading training radiomics...")

rad_train = pd.read_csv(
    RAD_TRAIN_FILE
)

print("Training radiomics:", rad_train.shape)

# ------------------------------------------------------------
# Load testing radiomics
# ------------------------------------------------------------

print("\nLoading testing radiomics...")

rad_test = pd.read_csv(
    RAD_TEST_FILE
)

print("Testing radiomics:", rad_test.shape)

# ------------------------------------------------------------
# Load testing CNN
# ------------------------------------------------------------

print("\nLoading testing CNN features...")

cnn_test = pd.read_csv(
    CNN_TEST_FILE
)

print("Testing CNN:", cnn_test.shape)

# ------------------------------------------------------------
# Load saved PCA
# ------------------------------------------------------------

print("\nLoading saved training PCA...")

pca = joblib.load(
    PCA_FILE
)

print("PCA loaded successfully!")

print(
    "PCA components:",
    pca.n_components_
)

# ------------------------------------------------------------
# Radiomics feature columns
# ------------------------------------------------------------

metadata_columns = [
    "image",
    "class"
]

feature_columns = [
    col
    for col in rad_train.columns
    if col not in metadata_columns
]

print(
    "\nRadiomics feature count:",
    len(feature_columns)
)

# ------------------------------------------------------------
# Missing values
# ------------------------------------------------------------

print("\nChecking missing values...")

train_missing = (
    rad_train[feature_columns]
    .isnull()
    .sum()
    .sum()
)

test_missing = (
    rad_test[feature_columns]
    .isnull()
    .sum()
    .sum()
)

print(
    "Training missing values:",
    train_missing
)

print(
    "Testing missing values:",
    test_missing
)

# Use training medians
train_medians = (
    rad_train[feature_columns]
    .median()
)

rad_train[feature_columns] = (
    rad_train[feature_columns]
    .fillna(train_medians)
)

rad_test[feature_columns] = (
    rad_test[feature_columns]
    .fillna(train_medians)
)

# ------------------------------------------------------------
# SCALE RADIOMICS
# ------------------------------------------------------------

print("\nScaling radiomics features...")

radiomics_scaler = StandardScaler()

X_rad_train = (
    radiomics_scaler
    .fit_transform(
        rad_train[feature_columns]
    )
)

X_rad_test = (
    radiomics_scaler
    .transform(
        rad_test[feature_columns]
    )
)

print(
    "Scaled training radiomics:",
    X_rad_train.shape
)

print(
    "Scaled testing radiomics:",
    X_rad_test.shape
)

# ------------------------------------------------------------
# PCA
# USE THE SAME PCA FIT ON TRAINING DATA
# ------------------------------------------------------------

print("\nApplying SAVED training PCA...")

X_pca_test = pca.transform(
    X_rad_test
)

print(
    "Testing PCA shape:",
    X_pca_test.shape
)

# ------------------------------------------------------------
# Create TEST PCA dataframe
# ------------------------------------------------------------

pca_columns = [
    f"PC{i+1}"
    for i in range(
        X_pca_test.shape[1]
    )
]

pca_test_df = pd.DataFrame(
    X_pca_test,
    columns=pca_columns
)

pca_test_df.insert(
    0,
    "class",
    rad_test["class"].values
)

pca_test_df.insert(
    0,
    "image",
    rad_test["image"].values
)

print(
    "\nRadiomics PCA test dataframe:",
    pca_test_df.shape
)

# ------------------------------------------------------------
# MERGE CNN + RADIOMICS
# ------------------------------------------------------------

print(
    "\nMatching CNN and radiomics using image filename..."
)

hybrid_test = pd.merge(
    cnn_test,
    pca_test_df,
    on="image",
    how="inner",
    suffixes=(
        "_cnn",
        "_radiomics"
    )
)

print(
    "Hybrid test shape after merge:",
    hybrid_test.shape
)

# ------------------------------------------------------------
# Check missing images
# ------------------------------------------------------------

print("\nChecking image coverage...")

cnn_images = set(
    cnn_test["image"]
)

rad_images = set(
    rad_test["image"]
)

missing_images = (
    cnn_images - rad_images
)

print(
    "CNN test images:",
    len(cnn_images)
)

print(
    "Radiomics test images:",
    len(rad_images)
)

print(
    "Images without radiomics:",
    len(missing_images)
)

if len(missing_images) > 0:
    print(
        "\nWARNING:",
        len(missing_images),
        "test images could not be included."
    )

# ------------------------------------------------------------
# CLASS CONSISTENCY
# ------------------------------------------------------------

print("\nChecking class consistency...")

if (
    "class_cnn" in hybrid_test.columns
    and
    "class_radiomics" in hybrid_test.columns
):

    mismatch = (
        hybrid_test["class_cnn"]
        != hybrid_test["class_radiomics"]
    ).sum()

    print(
        "Class mismatches:",
        mismatch
    )

    if mismatch > 0:
        print(
            "ERROR: Class labels do not match!"
        )
        exit()

    hybrid_test["class"] = (
        hybrid_test["class_cnn"]
    )

    hybrid_test.drop(
        columns=[
            "class_cnn",
            "class_radiomics"
        ],
        inplace=True
    )

# ------------------------------------------------------------
# SAVE
# ------------------------------------------------------------

os.makedirs(
    os.path.dirname(OUTPUT_FILE),
    exist_ok=True
)

hybrid_test.to_csv(
    OUTPUT_FILE,
    index=False
)

# ------------------------------------------------------------
# FINAL INFORMATION
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("TEST HYBRID FEATURE CREATION COMPLETED!")
print("=" * 60)

print("\nOutput:")
print(
    os.path.abspath(
        OUTPUT_FILE
    )
)

print(
    "\nFinal shape:",
    hybrid_test.shape
)

print(
    "\nClass distribution:"
)

print(
    hybrid_test["class"]
    .value_counts()
)

print(
    "\nNumber of features excluding image/class:"
)

print(
    hybrid_test
    .drop(
        columns=[
            "image",
            "class"
        ]
    )
    .shape[1]
)

print("\nFirst 5 rows:")

print(
    hybrid_test
    .iloc[:, :10]
    .head()
)
