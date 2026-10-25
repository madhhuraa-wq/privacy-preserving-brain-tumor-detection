import os
import cv2
import numpy as np

# ==============================
# SETTINGS
# ==============================

DATASET_PATH = "../dataset"
IMG_SIZE = 128

# ==============================
# CHECK DATASET
# ==============================

def check_dataset(folder_path):

    class_names = sorted(os.listdir(folder_path))

    print("Classes found:", class_names)

    total_images = 0

    for class_name in class_names:

        class_path = os.path.join(folder_path, class_name)

        if not os.path.isdir(class_path):
            continue

        count = 0

        for image_name in os.listdir(class_path):

            image_path = os.path.join(class_path, image_name)

            image = cv2.imread(image_path)

            if image is not None:
                count += 1

        print(f"{class_name}: {count} images")

        total_images += count

    print("\nTotal images:", total_images)

    return class_names


# ==============================
# CHECK TRAINING DATA
# ==============================

train_path = os.path.join(DATASET_PATH, "Training")

class_names = check_dataset(train_path)

# ==============================
# SAVE CLASS NAMES
# ==============================

os.makedirs("../models", exist_ok=True)

with open("../models/class_names.txt", "w") as file:

    for class_name in class_names:
        file.write(class_name + "\n")

print("\nClass names saved successfully!")

print("\nPreprocessing check completed successfully!")