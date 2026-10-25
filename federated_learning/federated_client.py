import os
import sys
import cv2
import numpy as np
import tensorflow as tf
import flwr as fl

# ============================================================
# FUZZY-AWARE FEDERATED LEARNING CLIENT
# ============================================================

IMG_SIZE = 64
NUM_CLASSES = 4

CLASS_NAMES = [
    "glioma",
    "meningioma",
    "notumor",
    "pituitary"
]

# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        ".."
    )
)

DATASET_PATH = os.path.join(
    PROJECT_ROOT,
    "dataset",
    "Training"
)

MODELS_PATH = os.path.join(
    PROJECT_ROOT,
    "models"
)

FUZZY_LABELS_PATH = os.path.join(
    MODELS_PATH,
    "fuzzy_cluster_labels.npy"
)

FUZZY_MEMBERSHIP_PATH = os.path.join(
    MODELS_PATH,
    "fuzzy_membership.npy"
)

# ============================================================
# CLIENT ID
# ============================================================

if len(sys.argv) < 2:

    print("ERROR: Client ID required.")
    print(
        "Example: python federated_learning/federated_client.py 0"
    )

    sys.exit(1)

CLIENT_ID = int(sys.argv[1])

if CLIENT_ID < 0 or CLIENT_ID >= NUM_CLASSES:

    print(
        "ERROR: Client ID must be 0, 1, 2, or 3."
    )

    sys.exit(1)

print("\n" + "=" * 60)
print("FUZZY-AWARE FEDERATED CLIENT")
print("=" * 60)

print(
    f"Client ID: {CLIENT_ID}"
)

# ============================================================
# CHECK PATHS
# ============================================================

print("\nDataset:")
print(DATASET_PATH)

if not os.path.exists(DATASET_PATH):

    print("\nERROR: Training dataset not found.")

    sys.exit(1)

if not os.path.exists(FUZZY_LABELS_PATH):

    print(
        "\nERROR: fuzzy_cluster_labels.npy not found."
    )

    sys.exit(1)

if not os.path.exists(FUZZY_MEMBERSHIP_PATH):

    print(
        "\nERROR: fuzzy_membership.npy not found."
    )

    sys.exit(1)

# ============================================================
# LOAD FUZZY RESULTS
# ============================================================

print("\nLoading fuzzy clustering results...")

cluster_labels = np.load(
    FUZZY_LABELS_PATH
)

membership = np.load(
    FUZZY_MEMBERSHIP_PATH
)

print(
    "Cluster labels:",
    cluster_labels.shape
)

print(
    "Membership:",
    membership.shape
)

# ============================================================
# FIND IMAGE PATHS
#
# IMPORTANT:
# Same ordering as fuzzy_clustering.py:
# sorted classes + sorted filenames
# ============================================================

print("\nFinding training images...")

all_image_paths = []
all_labels = []

for class_index, class_name in enumerate(
    CLASS_NAMES
):

    class_path = os.path.join(
        DATASET_PATH,
        class_name
    )

    if not os.path.isdir(class_path):

        print(
            "Missing folder:",
            class_path
        )

        continue

    print(
        f"Scanning {class_name}..."
    )

    for image_name in sorted(
        os.listdir(class_path)
    ):

        image_path = os.path.join(
            class_path,
            image_name
        )

        if not os.path.isfile(
            image_path
        ):
            continue

        image = cv2.imread(
            image_path
        )

        if image is None:
            continue

        all_image_paths.append(
            image_path
        )

        all_labels.append(
            class_index
        )

all_labels = np.array(
    all_labels,
    dtype=np.int32
)

print(
    "\nTotal valid training images:",
    len(all_image_paths)
)

# ============================================================
# SAFETY CHECK
# ============================================================

if len(all_image_paths) != len(
    cluster_labels
):

    print("\n" + "=" * 60)
    print("ERROR: IMAGE ORDER/COUNT MISMATCH")
    print("=" * 60)

    print(
        "Images:",
        len(all_image_paths)
    )

    print(
        "Fuzzy labels:",
        len(cluster_labels)
    )

    print(
        "\nRe-run fuzzy clustering before "
        "starting federated learning."
    )

    sys.exit(1)

# ============================================================
# SELECT FUZZY CLUSTER
# ============================================================

client_indices = np.where(
    cluster_labels == CLIENT_ID
)[0]

print("\n" + "-" * 60)
print("FUZZY CLIENT ASSIGNMENT")
print("-" * 60)

print(
    f"Client {CLIENT_ID} images:",
    len(client_indices)
)

client_membership = membership[
    CLIENT_ID,
    client_indices
]

print(
    "Average membership:",
    f"{np.mean(client_membership):.4f}"
)

print(
    "Minimum membership:",
    f"{np.min(client_membership):.4f}"
)

print(
    "Maximum membership:",
    f"{np.max(client_membership):.4f}"
)

# ============================================================
# LOAD CLIENT DATA
# ============================================================

print(
    "\nLoading client images..."
)

X_client = []
y_client = []

for index in client_indices:

    image_path = all_image_paths[index]

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

    X_client.append(
        image
    )

    y_client.append(
        all_labels[index]
    )

X_client = np.array(
    X_client,
    dtype=np.float32
)

y_client = np.array(
    y_client,
    dtype=np.int32
)

print("\nClient dataset:")
print(
    "X:",
    X_client.shape
)

print(
    "y:",
    y_client.shape
)

# ============================================================
# CLIENT CLASS DISTRIBUTION
# ============================================================

print("\nClient class distribution:")

for class_index, class_name in enumerate(
    CLASS_NAMES
):

    count = np.sum(
        y_client == class_index
    )

    print(
        f"{class_name}: {count}"
    )

# ============================================================
# CREATE MODEL
# ============================================================

def create_model():

    model = tf.keras.Sequential([

        tf.keras.layers.Input(
            shape=(64, 64, 3)
        ),

        tf.keras.layers.Conv2D(
            32,
            (3, 3),
            activation="relu"
        ),

        tf.keras.layers.MaxPooling2D(),

        tf.keras.layers.Conv2D(
            64,
            (3, 3),
            activation="relu"
        ),

        tf.keras.layers.MaxPooling2D(),

        tf.keras.layers.Conv2D(
            128,
            (3, 3),
            activation="relu"
        ),

        tf.keras.layers.MaxPooling2D(),

        tf.keras.layers.Flatten(),

        tf.keras.layers.Dense(
            128,
            activation="relu"
        ),

        tf.keras.layers.Dropout(
            0.5
        ),

        tf.keras.layers.Dense(
            NUM_CLASSES,
            activation="softmax"
        )
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


model = create_model()

# ============================================================
# FEDERATED CLIENT
# ============================================================

class BrainTumorClient(
    fl.client.NumPyClient
):

    def get_parameters(
        self,
        config
    ):

        return model.get_weights()

    def fit(
        self,
        parameters,
        config
    ):

        model.set_weights(
            parameters
        )

        print(
            f"\nClient {CLIENT_ID}: "
            "Starting local training..."
        )

        history = model.fit(

            X_client,

            y_client,

            epochs=2,

            batch_size=16,

            verbose=1,

            validation_split=0.1
        )

        accuracy = history.history[
            "accuracy"
        ][-1]

        loss = history.history[
            "loss"
        ][-1]

        print(
            f"\nClient {CLIENT_ID} training completed."
        )

        print(
            f"Local accuracy: {accuracy:.4f}"
        )

        print(
            f"Local loss: {loss:.4f}"
        )

        return (

            model.get_weights(),

            len(X_client),

            {
                "accuracy": float(
                    accuracy
                ),

                "loss": float(
                    loss
                )
            }
        )

    def evaluate(
        self,
        parameters,
        config
    ):

        model.set_weights(
            parameters
        )

        loss, accuracy = model.evaluate(
            X_client,
            y_client,
            verbose=0
        )

        print(
            f"\nClient {CLIENT_ID} evaluation:"
        )

        print(
            f"Loss: {loss:.4f}"
        )

        print(
            f"Accuracy: {accuracy:.4f}"
        )

        return (

            float(loss),

            len(X_client),

            {
                "accuracy": float(
                    accuracy
                )
            }
        )


# ============================================================
# START CLIENT
# ============================================================

print(
    "\nConnecting to Flower server..."
)

fl.client.start_numpy_client(
    server_address="127.0.0.1:8080",
    client=BrainTumorClient()
)