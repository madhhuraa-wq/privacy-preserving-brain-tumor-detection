import os
import cv2
import numpy as np
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# ============================================================
# SETTINGS
# ============================================================

IMG_SIZE = 64

CLASS_NAMES = [
    "glioma",
    "meningioma",
    "notumor",
    "pituitary"
]

BASE_PATH = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        ".."
    )
)

TEST_PATH = os.path.join(
    BASE_PATH,
    "dataset",
    "Testing"
)

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "federated_global_model.keras"
)

RESULTS_PATH = os.path.join(
    os.path.dirname(__file__),
    "results"
)

os.makedirs(RESULTS_PATH, exist_ok=True)

# ============================================================
# DISPLAY PATHS
# ============================================================

print("\n==========================================")
print("   FEDERATED MODEL EVALUATION")
print("==========================================")

print("\nTesting dataset:")
print(TEST_PATH)

print("\nFederated model:")
print(MODEL_PATH)

# ============================================================
# LOAD TESTING DATA
# ============================================================

print("\nLoading testing images...")

X_test = []
y_test = []

for class_index, class_name in enumerate(CLASS_NAMES):

    class_path = os.path.join(
        TEST_PATH,
        class_name
    )

    print(f"Loading {class_name}...")

    if not os.path.exists(class_path):
        print("Missing folder:", class_path)
        continue

    for image_name in sorted(
        os.listdir(class_path)
    ):

        image_path = os.path.join(
            class_path,
            image_name
        )

        image = cv2.imread(
            image_path
        )

        if image is None:
            continue

        image = cv2.resize(
            image,
            (IMG_SIZE, IMG_SIZE)
        )

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        image = image.astype(
            np.float32
        ) / 255.0

        X_test.append(image)

        y_test.append(
            class_index
        )

X_test = np.array(
    X_test,
    dtype=np.float32
)

y_test = np.array(
    y_test,
    dtype=np.int32
)

print("\nTotal test images:")
print(len(X_test))

print("Test shape:")
print(X_test.shape)

# ============================================================
# LOAD FEDERATED MODEL
# ============================================================

print("\nLoading federated model...")

model = tf.keras.models.load_model(
    MODEL_PATH
)

print("Model loaded successfully.")

# ============================================================
# PREDICTIONS
# ============================================================

print("\nRunning predictions...")

predictions = model.predict(
    X_test,
    batch_size=32,
    verbose=1
)

y_pred = np.argmax(
    predictions,
    axis=1
)

# ============================================================
# METRICS
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

print("\n==========================================")
print("       FEDERATED MODEL RESULTS")
print("==========================================")

print(
    f"\nAccuracy  : {accuracy * 100:.2f}%"
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

# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\n==========================================")
print("       CLASSIFICATION REPORT")
print("==========================================")

report = classification_report(
    y_test,
    y_pred,
    target_names=CLASS_NAMES,
    zero_division=0
)

print(report)

# ============================================================
# CONFUSION MATRIX
# ============================================================

print("\n==========================================")
print("          CONFUSION MATRIX")
print("==========================================")

cm = confusion_matrix(
    y_test,
    y_pred
)

print("\nRows = Actual")
print("Columns = Predicted\n")

print(
    "             ",
    CLASS_NAMES
)

for i, row in enumerate(cm):

    print(
        f"{CLASS_NAMES[i]:12}",
        row
    )

# ============================================================
# SAVE RESULTS
# ============================================================

results_file = os.path.join(
    RESULTS_PATH,
    "federated_evaluation.txt"
)

with open(
    results_file,
    "w"
) as f:

    f.write(
        "FEDERATED MODEL EVALUATION\n"
    )

    f.write(
        "===========================\n\n"
    )

    f.write(
        f"Accuracy  : {accuracy * 100:.2f}%\n"
    )

    f.write(
        f"Precision : {precision * 100:.2f}%\n"
    )

    f.write(
        f"Recall    : {recall * 100:.2f}%\n"
    )

    f.write(
        f"F1 Score  : {f1 * 100:.2f}%\n\n"
    )

    f.write(
        "CLASSIFICATION REPORT\n"
    )

    f.write(
        "=====================\n\n"
    )

    f.write(report)

    f.write(
        "\n\nCONFUSION MATRIX\n"
    )

    f.write(
        "================\n\n"
    )

    f.write(
        str(cm)
    )

print(
    "\nResults saved to:"
)

print(
    results_file
)

print("\n==========================================")
print("       EVALUATION COMPLETED")
print("==========================================")