import pandas as pd
import numpy as np
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

import joblib

# ============================================================
# HYBRID CNN + RADIOMICS MODEL
# ============================================================

INPUT_FILE = "features/hybrid_features.csv"

MODEL_FILE = "models/hybrid_random_forest.pkl"
SCALER_FILE = "models/hybrid_scaler.pkl"

print("=" * 60)
print("HYBRID CNN + RADIOMICS MODEL TRAINING")
print("=" * 60)

# ------------------------------------------------------------
# Load hybrid features
# ------------------------------------------------------------

print("\nLoading hybrid features...")

df = pd.read_csv(INPUT_FILE)

print("Dataset shape:", df.shape)

# ------------------------------------------------------------
# Separate features and labels
# ------------------------------------------------------------

X = df.drop(
    columns=["image", "class"]
)

y = df["class"]

print("\nFeature shape:", X.shape)
print("Number of features:", X.shape[1])

print("\nClass distribution:")
print(y.value_counts())

# ------------------------------------------------------------
# Train / validation split
# ------------------------------------------------------------

print("\nSplitting dataset...")

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training samples:", X_train.shape[0])
print("Validation samples:", X_val.shape[0])

# ------------------------------------------------------------
# Scale features
# ------------------------------------------------------------

print("\nScaling hybrid features...")

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_val_scaled = scaler.transform(X_val)

# ------------------------------------------------------------
# Train Random Forest
# ------------------------------------------------------------

print("\nTraining Random Forest...")

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced"
)

model.fit(
    X_train_scaled,
    y_train
)

print("Training completed!")

# ------------------------------------------------------------
# Validation prediction
# ------------------------------------------------------------

print("\nEvaluating hybrid model...")

y_pred = model.predict(X_val_scaled)

accuracy = accuracy_score(
    y_val,
    y_pred
)

print("\n" + "=" * 60)
print("HYBRID MODEL RESULTS")
print("=" * 60)

print(
    f"\nValidation Accuracy: {accuracy * 100:.2f}%"
)

# ------------------------------------------------------------
# Classification report
# ------------------------------------------------------------

print("\nClassification Report:")

print(
    classification_report(
        y_val,
        y_pred
    )
)

# ------------------------------------------------------------
# Confusion matrix
# ------------------------------------------------------------

print("\nConfusion Matrix:")

cm = confusion_matrix(
    y_val,
    y_pred
)

print(cm)

# ------------------------------------------------------------
# Save model
# ------------------------------------------------------------

os.makedirs(
    "../../models",
    exist_ok=True
)

joblib.dump(
    model,
    MODEL_FILE
)

joblib.dump(
    scaler,
    SCALER_FILE
)

print("\n" + "=" * 60)
print("MODEL SAVED")
print("=" * 60)

print("\nModel:")
print(os.path.abspath(MODEL_FILE))

print("\nScaler:")
print(os.path.abspath(SCALER_FILE))

print("\nHybrid training pipeline completed!")