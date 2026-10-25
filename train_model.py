import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# -----------------------------
# 1. SETTINGS
# -----------------------------

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20

TRAIN_DIR = "dataset/Training"
TEST_DIR = "dataset/Testing"

# -----------------------------
# 2. CHECK DATASET
# -----------------------------

print("Checking dataset...")

print("Training folder exists:", os.path.exists(TRAIN_DIR))
print("Testing folder exists:", os.path.exists(TEST_DIR))

# -----------------------------
# 3. LOAD TRAINING DATA
# -----------------------------

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.20,
    rotation_range=10,
    width_shift_range=0.10,
    height_shift_range=0.10,
    zoom_range=0.10,
    horizontal_flip=True
)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training",
    shuffle=True
)

validation_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation",
    shuffle=True
)

# -----------------------------
# 4. LOAD TEST DATA
# -----------------------------

test_datagen = ImageDataGenerator(
    rescale=1.0 / 255
)

test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

# -----------------------------
# 5. SHOW CLASS NAMES
# -----------------------------

print("\nClass names:")
print(train_generator.class_indices)

# -----------------------------
# 6. CREATE CNN MODEL
# -----------------------------

model = models.Sequential([
    
    layers.Input(shape=(224, 224, 3)),

    layers.Conv2D(32, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(128, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(256, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),

    layers.Dense(256, activation="relu"),
    layers.Dropout(0.5),

    layers.Dense(4, activation="softmax")
])

# -----------------------------
# 7. COMPILE MODEL
# -----------------------------

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# -----------------------------
# 8. DISPLAY MODEL
# -----------------------------

model.summary()

# -----------------------------
# 9. TRAIN MODEL
# -----------------------------

print("\nStarting training...\n")

history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=EPOCHS
)

# -----------------------------
# 10. TEST MODEL
# -----------------------------

print("\nEvaluating model on testing dataset...\n")

test_loss, test_accuracy = model.evaluate(test_generator)

print(f"\nTest Accuracy: {test_accuracy * 100:.2f}%")
print(f"Test Loss: {test_loss:.4f}")

# -----------------------------
# 11. SAVE MODEL
# -----------------------------

model.save("brain_tumor_model.keras")

print("\nModel saved successfully!")
print("File: brain_tumor_model.keras")
