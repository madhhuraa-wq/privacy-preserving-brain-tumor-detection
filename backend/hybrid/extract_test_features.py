import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model, Model

# ============================================================
# EXTRACT CNN FEATURES FROM TESTING DATA
# ============================================================

MODEL_PATH = "../../brain_tumor_model.keras"

IMAGE_DIR = "../../dataset/Testing"

OUTPUT_FILE = "../../features/cnn_testing_features.csv"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

print("=" * 60)
print("CNN TEST FEATURE EXTRACTION")
print("=" * 60)

# ------------------------------------------------------------
# Load trained CNN
# ------------------------------------------------------------

print("\nLoading trained CNN model...")

model = load_model(MODEL_PATH)

print("CNN model loaded successfully!")

print("Model input shape:", model.input_shape)
print("Model output shape:", model.output_shape)

# ------------------------------------------------------------
# Create feature extractor
# ------------------------------------------------------------

print("\nCreating CNN feature extractor...")

dummy_input = tf.zeros((1, 224, 224, 3))
_ = model(dummy_input)

feature_extractor = Model(
    inputs=model.inputs,
    outputs=model.layers[-2].output
)

print("Feature extractor created!")

# ------------------------------------------------------------
# Load testing dataset
# ------------------------------------------------------------

print("\nLoading testing images...")

dataset = tf.keras.utils.image_dataset_from_directory(
    IMAGE_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

class_names = dataset.class_names

print("\nClasses:", class_names)

# ------------------------------------------------------------
# Extract CNN features
# ------------------------------------------------------------

print("\nExtracting CNN test features...")

features = []
labels = []

for images, batch_labels in dataset:

    batch_features = feature_extractor.predict(
        images,
        verbose=0
    )

    features.append(batch_features)
    labels.append(batch_labels.numpy())

features = np.concatenate(features)
labels = np.concatenate(labels)

print("\nRaw feature shape:", features.shape)
print("Label shape:", labels.shape)

# ------------------------------------------------------------
# Flatten features
# ------------------------------------------------------------

if len(features.shape) > 2:

    features = features.reshape(
        features.shape[0],
        -1
    )

print("Final CNN feature shape:", features.shape)

# ------------------------------------------------------------
# Get filenames
# ------------------------------------------------------------

image_paths = dataset.file_paths

image_names = [
    os.path.basename(path)
    for path in image_paths
]

print("Number of image names:", len(image_names))

# ------------------------------------------------------------
# Create DataFrame
# ------------------------------------------------------------

feature_columns = [
    f"CNN_{i+1}"
    for i in range(features.shape[1])
]

cnn_df = pd.DataFrame(
    features,
    columns=feature_columns
)

cnn_df.insert(
    0,
    "image",
    image_names
)

cnn_df["class"] = [
    class_names[label]
    for label in labels
]

# ------------------------------------------------------------
# Save
# ------------------------------------------------------------

os.makedirs(
    os.path.dirname(OUTPUT_FILE),
    exist_ok=True
)

cnn_df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n" + "=" * 60)
print("CNN TEST FEATURE EXTRACTION COMPLETED!")
print("=" * 60)

print("Output file:")
print(os.path.abspath(OUTPUT_FILE))

print("Final shape:", cnn_df.shape)

print("\nClass distribution:")
print(cnn_df["class"].value_counts())

print("\nFirst 5 rows:")
print(cnn_df.iloc[:, :8].head())
