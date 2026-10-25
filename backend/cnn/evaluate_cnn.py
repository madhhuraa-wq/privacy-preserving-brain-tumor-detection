import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ============================================================
# CNN-ONLY TEST EVALUATION
# SAME PREPROCESSING AS ORIGINAL TRAINING
# ============================================================

PROJECT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

MODEL_PATH = os.path.join(
    PROJECT_DIR,
    "brain_tumor_model.keras"
)

TEST_DIR = os.path.join(
    PROJECT_DIR,
    "dataset",
    "Testing"
)

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

print("=" * 60)
print("CNN-ONLY TEST EVALUATION")
print("=" * 60)

# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading trained CNN model...")

print("Model:")
print(MODEL_PATH)

model = load_model(MODEL_PATH)

print("\nModel loaded successfully!")

print("Input shape :", model.input_shape)
print("Output shape:", model.output_shape)

# ============================================================
# LOAD TEST DATA
# SAME AS train_model.py
# ============================================================

print("\nLoading test dataset...")

test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1.0 / 255
)

test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

print("\nClass indices:")
print(test_generator.class_indices)

# ============================================================
# CLASS ORDER
# ============================================================

class_names = list(test_generator.class_indices.keys())

print("\nClass names:")
print(class_names)

# ============================================================
# EVALUATE MODEL
# ============================================================

print("\nEvaluating CNN on test dataset...")

test_loss, test_accuracy = model.evaluate(
    test_generator,
    verbose=1
)

print("\n" + "=" * 60)
print("CNN TEST ACCURACY")
print("=" * 60)

print(
    f"\nTest Accuracy: {test_accuracy * 100:.2f}%"
)

print(
    f"Test Loss: {test_loss:.4f}"
)

# ============================================================
# PREDICTIONS
# ============================================================

print("\nMaking predictions...")

test_generator.reset()

probabilities = model.predict(
    test_generator,
    verbose=1
)

y_pred = np.argmax(
    probabilities,
    axis=1
)

y_true = test_generator.classes

# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)

print(
    classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        digits=2,
        zero_division=0
    )
)

# ============================================================
# CONFUSION MATRIX
# ============================================================

print("Confusion Matrix:")

cm = confusion_matrix(
    y_true,
    y_pred
)

print(cm)

# ============================================================
# DATASET SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("TEST DATASET SUMMARY")
print("=" * 60)

print("\nTotal test images:", len(y_true))

for class_index, class_name in enumerate(class_names):

    count = np.sum(
        y_true == class_index
    )

    print(
        f"{class_name}: {count}"
    )

# ============================================================
# COMPLETED
# ============================================================

print("\n" + "=" * 60)
print("CNN EVALUATION COMPLETED")
print("=" * 60)