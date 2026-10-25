import os
import sys
import cv2
import numpy as np
import tensorflow as tf

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

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "federated_global_model.keras"
)

# ============================================================
# CHECK IMAGE ARGUMENT
# ============================================================

if len(sys.argv) < 2:
    print("\nERROR: Please provide an image path.")
    print(
        r'Example: python predict.py "C:\path\to\image.jpg"'
    )
    sys.exit(1)

IMAGE_PATH = sys.argv[1]

# ============================================================
# DISPLAY INFORMATION
# ============================================================

print("\n==========================================")
print("   BRAIN TUMOR FEDERATED PREDICTION")
print("==========================================")

print("\nModel:")
print(MODEL_PATH)

print("\nImage:")
print(IMAGE_PATH)

# ============================================================
# CHECK FILES
# ============================================================

if not os.path.exists(MODEL_PATH):
    print("\nERROR: Federated model not found.")
    sys.exit(1)

if not os.path.exists(IMAGE_PATH):
    print("\nERROR: Image not found.")
    sys.exit(1)

# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading federated model...")

model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)

print("Model loaded successfully.")

# ============================================================
# LOAD IMAGE
# ============================================================

print("\nLoading MRI image...")

image = cv2.imread(
    IMAGE_PATH
)

if image is None:
    print("\nERROR: Could not read image.")
    sys.exit(1)

# ============================================================
# PREPROCESS
# ============================================================

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

image = np.expand_dims(
    image,
    axis=0
)

print(
    "Image prepared:",
    image.shape
)

# ============================================================
# PREDICTION
# ============================================================

print("\nRunning prediction...")

prediction = model.predict(
    image,
    verbose=0
)

predicted_class = np.argmax(
    prediction[0]
)

confidence = prediction[0][
    predicted_class
] * 100

# ============================================================
# DISPLAY ALL PROBABILITIES
# ============================================================

print("\n==========================================")
print("          PREDICTION RESULT")
print("==========================================")

print(
    f"\nPredicted Class : "
    f"{CLASS_NAMES[predicted_class]}"
)

print(
    f"Confidence      : "
    f"{confidence:.2f}%"
)

print("\nClass probabilities:")

for i, class_name in enumerate(CLASS_NAMES):

    print(
        f"{class_name:12} : "
        f"{prediction[0][i] * 100:.2f}%"
    )

print("\n==========================================")
print("             COMPLETE")
print("==========================================")