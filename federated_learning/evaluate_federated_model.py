import tensorflow as tf
import numpy as np
import os
from sklearn.metrics import classification_report, confusion_matrix

# ============================================================
# SETTINGS
# ============================================================

IMG_SIZE = (64, 64)
BATCH_SIZE = 32

CLASS_NAMES = [
    "glioma",
    "meningioma",
    "notumor",
    "pituitary"
]

BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "federated_global_model.keras"
)

TEST_PATH = os.path.join(
    BASE_PATH,
    "dataset",
    "Testing"
)

print("\n==========================================")
print("FEDERATED MODEL EVALUATION")
print("==========================================")

print("\nModel:")
print(MODEL_PATH)

print("\nTesting dataset:")
print(TEST_PATH)

# ============================================================
# CHECK PATHS
# ============================================================

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Federated model not found:\n{MODEL_PATH}"
    )

if not os.path.exists(TEST_PATH):
    raise FileNotFoundError(
        f"Testing dataset not found:\n{TEST_PATH}"
    )

# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading federated global model...")

model = tf.keras.models.load_model(
    MODEL_PATH
)

print("Model loaded successfully.")
print("Input shape :", model.input_shape)
print("Output shape:", model.output_shape)

# ============================================================
# LOAD TEST DATASET EFFICIENTLY
# ============================================================

print("\nLoading testing dataset...")

test_dataset = tf.keras.utils.image_dataset_from_directory(
    TEST_PATH,
    labels="inferred",
    label_mode="int",
    class_names=CLASS_NAMES,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

print("\nClasses:")
print(test_dataset.class_names)

# Normalize images
test_dataset = test_dataset.map(
    lambda images, labels: (
        tf.cast(images, tf.float32) / 255.0,
        labels
    )
)

# ============================================================
# EVALUATION
# ============================================================

print("\n==========================================")
print("TESTING FEDERATED GLOBAL MODEL")
print("==========================================")

loss, accuracy = model.evaluate(
    test_dataset,
    verbose=1
)

print(
    f"\nTest Loss: {loss:.4f}"
)

print(
    f"Test Accuracy: {accuracy * 100:.2f}%"
)

# ============================================================
# PREDICTIONS
# ============================================================

print("\nGenerating predictions...")

predictions = model.predict(
    test_dataset,
    verbose=1
)

y_pred = np.argmax(
    predictions,
    axis=1
)

# ============================================================
# GET TRUE LABELS
# ============================================================

y_test = np.concatenate(
    [
        labels.numpy()
        for _, labels in test_dataset
    ]
)

print(
    f"\nTotal test images: {len(y_test)}"
)

# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\n==========================================")
print("CLASSIFICATION REPORT")
print("==========================================")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0
    )
)

# ============================================================
# CONFUSION MATRIX
# ============================================================

print("\n==========================================")
print("CONFUSION MATRIX")
print("==========================================")

cm = confusion_matrix(
    y_test,
    y_pred
)

print(cm)

# ============================================================
# CLASS DISTRIBUTION
# ============================================================

print("\n==========================================")
print("TEST CLASS DISTRIBUTION")
print("==========================================")

for i, class_name in enumerate(CLASS_NAMES):

    actual_count = np.sum(y_test == i)
    predicted_count = np.sum(y_pred == i)

    print(
        f"{class_name}: "
        f"Actual={actual_count}, "
        f"Predicted={predicted_count}"
    )

# ============================================================
# COMPLETE
# ============================================================

print("\n==========================================")
print("FEDERATED EVALUATION COMPLETE")
print("==========================================")