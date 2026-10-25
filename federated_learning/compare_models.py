import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd

# ============================================================
# BRAIN TUMOR MODEL COMPARISON
# Original CNN vs Federated CNN
# ============================================================

print("\n==========================================")
print("       BRAIN TUMOR MODEL COMPARISON")
print("==========================================\n")

# ------------------------------------------------------------
# PATHS
# ------------------------------------------------------------

TEST_DIR = r"C:\Users\MADHURA M\OneDrive\Desktop\BrainTumorProject\dataset\Testing"

ORIGINAL_MODEL = r"C:\Users\MADHURA M\OneDrive\Desktop\BrainTumorProject\brain_tumor_model.keras"

FEDERATED_MODEL = r"C:\Users\MADHURA M\OneDrive\Desktop\BrainTumorProject\federated_learning\federated_global_model.keras"

# ------------------------------------------------------------
# CLASS NAMES
# ------------------------------------------------------------

CLASS_NAMES = [
    "glioma",
    "meningioma",
    "notumor",
    "pituitary"
]

print("Testing dataset:")
print(TEST_DIR)

print("\nOriginal CNN:")
print(ORIGINAL_MODEL)

print("\nFederated CNN:")
print(FEDERATED_MODEL)

# ------------------------------------------------------------
# LOAD DATASET
# ------------------------------------------------------------

print("\nLoading testing images...")

image_paths = []
true_labels = []

for class_index, class_name in enumerate(CLASS_NAMES):

    class_folder = os.path.join(TEST_DIR, class_name)

    for filename in os.listdir(class_folder):

        if filename.lower().endswith((".jpg", ".jpeg", ".png")):

            image_paths.append(
                os.path.join(class_folder, filename)
            )

            true_labels.append(class_index)

true_labels = np.array(true_labels)

print("\nTotal test images:", len(image_paths))

# ------------------------------------------------------------
# FUNCTION TO LOAD IMAGES
# ------------------------------------------------------------

def load_images(image_size):

    images = []

    print(
        f"\nPreparing images for model input size "
        f"{image_size} x {image_size}..."
    )

    for path in image_paths:

        image = tf.keras.utils.load_img(
            path,
            target_size=(image_size, image_size)
        )

        image = tf.keras.utils.img_to_array(image)

        image = image / 255.0

        images.append(image)

    return np.array(images, dtype=np.float32)


# ------------------------------------------------------------
# FUNCTION TO EVALUATE MODEL
# ------------------------------------------------------------

def evaluate_model(model_path, image_size, model_name):

    print("\n------------------------------------------")
    print(model_name)
    print("------------------------------------------")

    print("Loading model...")

    model = tf.keras.models.load_model(
        model_path,
        compile=False
    )

    print("Model loaded successfully.")

    X_test = load_images(image_size)

    print("\nRunning predictions...")

    probabilities = model.predict(
        X_test,
        batch_size=32,
        verbose=1
    )

    predictions = np.argmax(
        probabilities,
        axis=1
    )

    accuracy = accuracy_score(
        true_labels,
        predictions
    )

    precision = precision_score(
        true_labels,
        predictions,
        average="macro",
        zero_division=0
    )

    recall = recall_score(
        true_labels,
        predictions,
        average="macro",
        zero_division=0
    )

    f1 = f1_score(
        true_labels,
        predictions,
        average="macro",
        zero_division=0
    )

    print("\nResults:")

    print(
        f"Accuracy  : {accuracy * 100:.2f}%"
    )

    print(
        f"Precision : {precision * 100:.2f}%"
    )

    print(
        f"Recall    : {recall * 100:.2f}%"
    )

    print(
        f"F1 Score  : {f1 * 100:.2f}%"
    )

    return {
        "Model": model_name,
        "Accuracy (%)": accuracy * 100,
        "Precision (%)": precision * 100,
        "Recall (%)": recall * 100,
        "F1 Score (%)": f1 * 100
    }


# ------------------------------------------------------------
# ORIGINAL CNN
# ------------------------------------------------------------

original_results = evaluate_model(
    ORIGINAL_MODEL,
    224,
    "Original CNN"
)


# ------------------------------------------------------------
# FEDERATED CNN
# ------------------------------------------------------------

federated_results = evaluate_model(
    FEDERATED_MODEL,
    64,
    "Federated CNN"
)


# ------------------------------------------------------------
# CREATE COMPARISON TABLE
# ------------------------------------------------------------

results = [
    original_results,
    federated_results
]

comparison = pd.DataFrame(results)

print("\n==========================================")
print("           MODEL COMPARISON")
print("==========================================\n")

print(
    comparison.to_string(index=False)
)

# ------------------------------------------------------------
# SAVE RESULTS
# ------------------------------------------------------------

RESULTS_DIR = os.path.join(
    os.path.dirname(__file__),
    "results"
)

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)

output_file = os.path.join(
    RESULTS_DIR,
    "model_comparison.csv"
)

comparison.to_csv(
    output_file,
    index=False
)

print("\n==========================================")
print("Comparison saved successfully!")
print("==========================================")

print("\nFile:")
print(output_file)

print("\nModel comparison completed.")