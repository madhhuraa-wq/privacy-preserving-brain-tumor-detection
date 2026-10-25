import os
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# ============================================================
# FINAL FUZZY + PCA + FEDERATED RADIOMICS
# ============================================================

print("=" * 65)
print("FINAL FUZZY + PCA + FEDERATED RADIOMICS")
print("=" * 65)

# ============================================================
# PATHS
# ============================================================

PROJECT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
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

FUZZY_LABELS_FILE = os.path.join(
    PROJECT_DIR,
    "models",
    "fuzzy_cluster_labels.npy"
)

MODEL_DIR = os.path.join(
    PROJECT_DIR,
    "models"
)

FINAL_MODEL_FILE = os.path.join(
    MODEL_DIR,
    "federated_radiomics_model.keras"
)

SCALER_FILE = os.path.join(
    MODEL_DIR,
    "federated_radiomics_scaler.pkl"
)

PCA_FILE = os.path.join(
    MODEL_DIR,
    "federated_radiomics_pca.pkl"
)

os.makedirs(MODEL_DIR, exist_ok=True)

# ============================================================
# SETTINGS
# ============================================================

N_COMPONENTS = 12
N_CLIENTS = 4
FEDERATED_ROUNDS = 4
LOCAL_EPOCHS = 3
BATCH_SIZE = 32

CLASS_NAMES = [
    "glioma",
    "meningioma",
    "notumor",
    "pituitary"
]

# ============================================================
# LOAD RADIOMICS DATA
# ============================================================

print("\nLoading radiomics data...")

train_df = pd.read_csv(TRAIN_FILE)
test_df = pd.read_csv(TEST_FILE)

feature_columns = [
    col for col in train_df.columns
    if col not in ["image", "class"]
]

X_train = train_df[feature_columns].values.astype(np.float32)
y_train_text = train_df["class"].values

X_test = test_df[feature_columns].values.astype(np.float32)
y_test_text = test_df["class"].values

# Convert class names to numbers
class_to_id = {
    "glioma": 0,
    "meningioma": 1,
    "notumor": 2,
    "pituitary": 3
}

y_train = np.array(
    [class_to_id[x] for x in y_train_text],
    dtype=np.int32
)

y_test = np.array(
    [class_to_id[x] for x in y_test_text],
    dtype=np.int32
)

print("Training:", X_train.shape)
print("Testing :", X_test.shape)
print("Radiomics features:", len(feature_columns))

# ============================================================
# SCALE FEATURES
# ============================================================

print("\nScaling radiomics features...")

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ============================================================
# PCA
# ============================================================

print("\nApplying PCA...")

pca = PCA(
    n_components=N_COMPONENTS,
    random_state=42
)

X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

explained_variance = np.sum(
    pca.explained_variance_ratio_
)

print("PCA training:", X_train_pca.shape)
print("PCA testing :", X_test_pca.shape)
print(
    "Explained variance:",
    round(explained_variance, 4)
)

# ============================================================
# LOAD FUZZY CLUSTERING RESULTS
# ============================================================

print("\nLoading fuzzy cluster assignments...")

if not os.path.exists(FUZZY_LABELS_FILE):
    raise FileNotFoundError(
        f"Fuzzy labels not found:\n{FUZZY_LABELS_FILE}"
    )

fuzzy_labels = np.load(
    FUZZY_LABELS_FILE
)

print("Fuzzy labels:", fuzzy_labels.shape)

if len(fuzzy_labels) != len(X_train_pca):
    raise ValueError(
        "Number of fuzzy labels does not match "
        "number of training samples."
    )

# ============================================================
# CREATE FEDERATED CLIENTS
# ============================================================

print("\nFederated client distribution:")

client_indices = []

for client_id in range(N_CLIENTS):

    indices = np.where(
        fuzzy_labels == client_id
    )[0]

    client_indices.append(indices)

    print(
        f"Client {client_id}: "
        f"{len(indices)} samples"
    )

# ============================================================
# CREATE MODEL
# ============================================================

def create_model():

    model = tf.keras.Sequential([
        tf.keras.layers.Input(
            shape=(N_COMPONENTS,)
        ),

        tf.keras.layers.Dense(
            64,
            activation="relu"
        ),

        tf.keras.layers.Dropout(0.2),

        tf.keras.layers.Dense(
            32,
            activation="relu"
        ),

        tf.keras.layers.Dense(
            4,
            activation="softmax"
        )
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.001
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


# ============================================================
# INITIAL GLOBAL MODEL
# ============================================================

global_model = create_model()

print("\nInitial global model created.")

# ============================================================
# FEDERATED AVERAGING
# ============================================================

def federated_average(
    client_weights,
    client_sizes
):

    total_samples = sum(client_sizes)

    averaged_weights = []

    for layer_weights in zip(*client_weights):

        weighted_sum = np.zeros_like(
            layer_weights[0]
        )

        for weights, size in zip(
            layer_weights,
            client_sizes
        ):

            weighted_sum += (
                weights * size / total_samples
            )

        averaged_weights.append(
            weighted_sum
        )

    return averaged_weights


# ============================================================
# FEDERATED TRAINING
# ============================================================

for round_number in range(
    1,
    FEDERATED_ROUNDS + 1
):

    print("\n" + "=" * 65)
    print(
        f"FEDERATED ROUND "
        f"{round_number}/{FEDERATED_ROUNDS}"
    )
    print("=" * 65)

    global_weights = global_model.get_weights()

    client_weights = []
    client_sizes = []

    for client_id in range(N_CLIENTS):

        indices = client_indices[client_id]

        if len(indices) == 0:
            print(
                f"\nClient {client_id}: EMPTY"
            )
            continue

        print(
            f"\nClient {client_id}: "
            f"{len(indices)} samples"
        )

        X_client = X_train_pca[indices]
        y_client = y_train[indices]

        # Create local model
        local_model = create_model()

        # Start from global model
        local_model.set_weights(
            global_weights
        )

        # Local training
        local_model.fit(
            X_client,
            y_client,
            epochs=LOCAL_EPOCHS,
            batch_size=BATCH_SIZE,
            verbose=0,
            shuffle=True
        )

        # Local accuracy
        local_loss, local_accuracy = (
            local_model.evaluate(
                X_client,
                y_client,
                verbose=0
            )
        )

        print(
            f"Client {client_id} accuracy: "
            f"{local_accuracy:.4f}"
        )

        client_weights.append(
            local_model.get_weights()
        )

        client_sizes.append(
            len(indices)
        )

        del local_model

    # --------------------------------------------------------
    # FEDAVG
    # --------------------------------------------------------

    print("\nPerforming FedAvg aggregation...")

    new_global_weights = federated_average(
        client_weights,
        client_sizes
    )

    global_model.set_weights(
        new_global_weights
    )

    # --------------------------------------------------------
    # GLOBAL TEST
    # --------------------------------------------------------

    global_loss, global_accuracy = (
        global_model.evaluate(
            X_test_pca,
            y_test,
            verbose=0
        )
    )

    print(
        f"\nGlobal test accuracy after "
        f"round {round_number}: "
        f"{global_accuracy:.4f}"
    )

# ============================================================
# FINAL FEDERATED TEST
# ============================================================

print("\n" + "=" * 65)
print("FINAL FEDERATED RADIOMICS TEST")
print("=" * 65)

final_loss, final_accuracy = (
    global_model.evaluate(
        X_test_pca,
        y_test,
        verbose=0
    )
)

print(
    f"\nTest Loss: {final_loss:.4f}"
)

print(
    f"Test Accuracy: "
    f"{final_accuracy * 100:.2f}%"
)

# ============================================================
# PREDICTIONS
# ============================================================

print("\nGenerating predictions...")

probabilities = global_model.predict(
    X_test_pca,
    verbose=0
)

y_pred = np.argmax(
    probabilities,
    axis=1
)

# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 65)
print("CLASSIFICATION REPORT")
print("=" * 65)

print(
    classification_report(
        y_test,
        y_pred,
        target_names=CLASS_NAMES,
        digits=4
    )
)

# ============================================================
# CONFUSION MATRIX
# ============================================================

print("\n" + "=" * 65)
print("CONFUSION MATRIX")
print("=" * 65)

cm = confusion_matrix(
    y_test,
    y_pred
)

print(cm)

# ============================================================
# CLASS-WISE RESULTS
# ============================================================

print("\n" + "=" * 65)
print("CLASS-WISE RESULTS")
print("=" * 65)

for i, class_name in enumerate(CLASS_NAMES):

    actual = np.sum(y_test == i)
    predicted = np.sum(y_pred == i)

    correct = cm[i, i]

    class_accuracy = (
        correct / actual
        if actual > 0
        else 0
    )

    print(
        f"{class_name}: "
        f"Actual={actual}, "
        f"Predicted={predicted}, "
        f"Correct={correct}, "
        f"Accuracy={class_accuracy * 100:.2f}%"
    )

# ============================================================
# SAVE FINAL MODEL
# ============================================================

print("\n" + "=" * 65)
print("SAVING FINAL MODEL")
print("=" * 65)

global_model.save(
    FINAL_MODEL_FILE
)

print(
    "\nFederated model saved:"
)

print(
    FINAL_MODEL_FILE
)

# ============================================================
# SAVE SCALER + PCA
# ============================================================

import joblib

joblib.dump(
    scaler,
    SCALER_FILE
)

joblib.dump(
    pca,
    PCA_FILE
)

print(
    "\nScaler saved:"
)

print(
    SCALER_FILE
)

print(
    "\nPCA saved:"
)

print(
    PCA_FILE
)

# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 65)
print("PIPELINE COMPLETED SUCCESSFULLY")
print("=" * 65)

print("\nPipeline:")
print(
    "MRI Images"
    " -> Radiomics"
    " -> Fuzzy Clustering"
    " -> PCA"
    " -> Federated Learning"
    " -> Classification"
)

print(
    f"\nRadiomics features: "
    f"{len(feature_columns)}"
)

print(
    f"PCA components: "
    f"{N_COMPONENTS}"
)

print(
    f"PCA explained variance: "
    f"{explained_variance * 100:.2f}%"
)

print(
    f"Federated clients: "
    f"{N_CLIENTS}"
)

print(
    f"Federated rounds: "
    f"{FEDERATED_ROUNDS}"
)

print(
    f"Final test accuracy: "
    f"{final_accuracy * 100:.2f}%"
)

print("\n" + "=" * 65)
print("DONE")
print("=" * 65)