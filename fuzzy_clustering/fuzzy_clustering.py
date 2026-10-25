import os
import cv2
import numpy as np
import skfuzzy as fuzz

# ============================================================
# FUZZY C-MEANS CLUSTERING
# ============================================================

print("=" * 60)
print("FUZZY C-MEANS CLUSTERING")
print("=" * 60)

# ============================================================
# PROJECT PATHS
# ============================================================

# Project root:
# C:\Users\MADHURA M\OneDrive\Desktop\BrainTumorProject

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
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

IMG_SIZE = 64
N_CLUSTERS = 4

print("\nProject root:")
print(PROJECT_ROOT)

print("\nTraining dataset:")
print(DATASET_PATH)

# ============================================================
# CHECK DATASET
# ============================================================

if not os.path.exists(DATASET_PATH):

    print("\nERROR!")
    print("Training dataset was not found.")
    print("Expected path:")
    print(DATASET_PATH)

    raise SystemExit(1)

print("\nTraining dataset found!")

# ============================================================
# LOAD IMAGES
# ============================================================

def load_images(folder_path):

    images = []
    labels = []

    class_names = sorted(
        [
            name
            for name in os.listdir(folder_path)
            if os.path.isdir(
                os.path.join(folder_path, name)
            )
        ]
    )

    print("\nClasses:")
    print(class_names)

    for label, class_name in enumerate(class_names):

        class_path = os.path.join(
            folder_path,
            class_name
        )

        print(
            f"\nLoading {class_name}..."
        )

        image_files = sorted(
            os.listdir(class_path)
        )

        valid_count = 0

        for image_name in image_files:

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
                cv2.COLOR_BGR2GRAY
            )

            image = (
                image.astype(
                    np.float32
                ) / 255.0
            )

            images.append(image)
            labels.append(label)

            valid_count += 1

        print(
            f"Valid images: {valid_count}"
        )

    return (
        np.array(images),
        np.array(labels),
        class_names
    )


# ============================================================
# LOAD DATA
# ============================================================

X, y, class_names = load_images(
    DATASET_PATH
)

print("\n" + "=" * 60)
print("DATASET LOADED")
print("=" * 60)

print(
    "Images:",
    X.shape
)

print(
    "Labels:",
    y.shape
)

# ============================================================
# EXTRACT COMPACT FEATURES
# ============================================================

print("\nExtracting image features...")

features = []

for image in X:

    mean = np.mean(image)

    std = np.std(image)

    min_value = np.min(image)

    max_value = np.max(image)

    center = image[
        16:48,
        16:48
    ]

    center_mean = np.mean(
        center
    )

    center_std = np.std(
        center
    )

    features.append(
        [
            mean,
            std,
            min_value,
            max_value,
            center_mean,
            center_std
        ]
    )

features = np.array(
    features,
    dtype=np.float32
)

print(
    "Feature shape:",
    features.shape
)

# ============================================================
# NORMALIZE FEATURES
# ============================================================

print("\nNormalizing features...")

features_min = features.min(
    axis=0
)

features_max = features.max(
    axis=0
)

features = (
    features - features_min
) / (
    features_max -
    features_min +
    1e-8
)

# ============================================================
# FUZZY C-MEANS
# ============================================================

print("\n" + "=" * 60)
print("RUNNING FUZZY C-MEANS")
print("=" * 60)

data = features.T

centers, membership, _, _, _, _, _ = (
    fuzz.cluster.cmeans(
        data,
        c=N_CLUSTERS,
        m=2.0,
        error=0.001,
        maxiter=500,
        seed=42
    )
)

cluster_labels = np.argmax(
    membership,
    axis=0
)

# ============================================================
# RESULTS
# ============================================================

print("\nFuzzy clustering completed!")

print(
    "Cluster centers shape:",
    centers.shape
)

print(
    "Membership matrix shape:",
    membership.shape
)

print(
    "Cluster labels shape:",
    cluster_labels.shape
)

# ============================================================
# CLUSTER DISTRIBUTION
# ============================================================

print("\nCluster distribution:")

for cluster in range(
    N_CLUSTERS
):

    count = np.sum(
        cluster_labels == cluster
    )

    print(
        f"Cluster {cluster}: {count} images"
    )

# ============================================================
# SAVE RESULTS
# ============================================================

os.makedirs(
    MODELS_PATH,
    exist_ok=True
)

centers_path = os.path.join(
    MODELS_PATH,
    "fuzzy_centers.npy"
)

membership_path = os.path.join(
    MODELS_PATH,
    "fuzzy_membership.npy"
)

labels_path = os.path.join(
    MODELS_PATH,
    "fuzzy_cluster_labels.npy"
)

np.save(
    centers_path,
    centers
)

np.save(
    membership_path,
    membership
)

np.save(
    labels_path,
    cluster_labels
)

# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 60)
print("FUZZY CLUSTERING STAGE COMPLETED")
print("=" * 60)

print("\nSaved files:")

print(
    centers_path
)

print(
    membership_path
)

print(
    labels_path
)

print("\nDataset summary:")

for i, class_name in enumerate(
    class_names
):

    count = np.sum(
        y == i
    )

    print(
        f"{class_name}: {count}"
    )

print("\nDone!")