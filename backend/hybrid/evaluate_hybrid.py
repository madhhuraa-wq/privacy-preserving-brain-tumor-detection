import pandas as pd
import joblib
import os

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# ============================================================
# HYBRID CNN + RADIOMICS TEST EVALUATION
# ============================================================

TEST_FILE = "features/hybrid_testing_features.csv"

MODEL_FILE = "models/hybrid_random_forest.pkl"
SCALER_FILE = "models/hybrid_scaler.pkl"

print("=" * 60)
print("HYBRID CNN + RADIOMICS TEST EVALUATION")
print("=" * 60)

# ------------------------------------------------------------
# Load test data
# ------------------------------------------------------------

print("\nLoading test hybrid features...")

df = pd.read_csv(TEST_FILE)

print("Test dataset shape:", df.shape)

# ------------------------------------------------------------
# Separate features and labels
# ------------------------------------------------------------

X_test = df.drop(columns=["image", "class"])
y_test = df["class"]

print("\nTest feature shape:", X_test.shape)

print("\nTest class distribution:")
print(y_test.value_counts())

# ------------------------------------------------------------
# Load trained model
# ------------------------------------------------------------

print("\nLoading trained Random Forest...")

model = joblib.load(MODEL_FILE)

print("Model loaded successfully!")

# ------------------------------------------------------------
# Load trained scaler
# ------------------------------------------------------------

print("\nLoading trained scaler...")

scaler = joblib.load(SCALER_FILE)

print("Scaler loaded successfully!")

# ------------------------------------------------------------
# Scale test data
# ------------------------------------------------------------

print("\nScaling test features...")

X_test_scaled = scaler.transform(X_test)

print("Test features scaled!")

# ------------------------------------------------------------
# Prediction
# ------------------------------------------------------------

print("\nMaking predictions...")

y_pred = model.predict(X_test_scaled)

# ------------------------------------------------------------
# Accuracy
# ------------------------------------------------------------

accuracy = accuracy_score(y_test, y_pred)

print("\n" + "=" * 60)
print("FINAL HYBRID TEST RESULTS")
print("=" * 60)

print(f"\nTest Accuracy: {accuracy * 100:.2f}%")

# ------------------------------------------------------------
# Classification report
# ------------------------------------------------------------

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred
    )
)

# ------------------------------------------------------------
# Confusion matrix
# ------------------------------------------------------------

print("\nConfusion Matrix:")

cm = confusion_matrix(
    y_test,
    y_pred
)

print(cm)

print("\n" + "=" * 60)
print("HYBRID TEST EVALUATION COMPLETED")
print("=" * 60)