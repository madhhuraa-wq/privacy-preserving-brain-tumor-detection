import os
import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ============================================================
# RADIOMICS-ONLY MODEL
# ============================================================

PROJECT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)

TRAIN_FILE = os.path.join(
    PROJECT_DIR,
    "radiomics",
    "results",
    "radiomics_features.csv"
)

TEST_FILE = os.path.join(
    PROJECT_DIR,
    "radiomics",
    "results",
    "radiomics_testing_features.csv"
)

MODEL_DIR = os.path.join(
    PROJECT_DIR,
    "models",
    "radiomics"
)

os.makedirs(MODEL_DIR, exist_ok=True)

print("=" * 60)
print("RADIOMICS-ONLY MODEL")
print("=" * 60)

# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading training radiomics...")
train_df = pd.read_csv(TRAIN_FILE)

print("Training shape:", train_df.shape)

print("\nLoading testing radiomics...")
test_df = pd.read_csv(TEST_FILE)

print("Testing shape:", test_df.shape)

# ============================================================
# FEATURES
# ============================================================

exclude_columns = ["image", "class"]

feature_columns = [
    col for col in train_df.columns
    if col not in exclude_columns
]

X_train = train_df[feature_columns]
y_train = train_df["class"]

X_test = test_df[feature_columns]
y_test = test_df["class"]

print("\nRadiomics features:", len(feature_columns))

# ============================================================
# SCALING
# ============================================================

print("\nScaling radiomics features...")

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Scaled training:", X_train_scaled.shape)
print("Scaled testing:", X_test_scaled.shape)

# ============================================================
# PCA
# ============================================================

print("\nApplying PCA...")

pca = PCA(n_components=12)

X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

print("Training PCA:", X_train_pca.shape)
print("Testing PCA:", X_test_pca.shape)

print(
    "Explained variance:",
    pca.explained_variance_ratio_.sum()
)

# ============================================================
# RANDOM FOREST
# ============================================================

print("\nTraining Radiomics Random Forest...")

model = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced"
)

model.fit(X_train_pca, y_train)

print("Training completed!")

# ============================================================
# TEST
# ============================================================

print("\nEvaluating Radiomics-only model...")

y_pred = model.predict(X_test_pca)

accuracy = accuracy_score(y_test, y_pred)

print("\n" + "=" * 60)
print("RADIOMICS-ONLY TEST RESULT")
print("=" * 60)

print("\nTest Accuracy:", f"{accuracy * 100:.2f}%")

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred
    )
)

print("\nConfusion Matrix:")

print(
    confusion_matrix(
        y_test,
        y_pred
    )
)

# ============================================================
# SAVE MODEL
# ============================================================

model_path = os.path.join(
    MODEL_DIR,
    "radiomics_random_forest.pkl"
)

scaler_path = os.path.join(
    MODEL_DIR,
    "radiomics_scaler.pkl"
)

pca_path = os.path.join(
    MODEL_DIR,
    "radiomics_pca.pkl"
)

joblib.dump(model, model_path)
joblib.dump(scaler, scaler_path)
joblib.dump(pca, pca_path)

print("\nModel saved:")
print(model_path)

print("\nScaler saved:")
print(scaler_path)

print("\nPCA saved:")
print(pca_path)

print("\n" + "=" * 60)
print("RADIOMICS-ONLY PIPELINE COMPLETED")
print("=" * 60)