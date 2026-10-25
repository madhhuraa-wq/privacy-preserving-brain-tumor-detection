import os
import cv2
import numpy as np
import pandas as pd
import SimpleITK as sitk

from radiomics import featureextractor


# ============================================================
# PATHS
# ============================================================

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATASET_DIR = os.path.join(PROJECT_DIR, "dataset", "Training")

OUTPUT_DIR = os.path.join(PROJECT_DIR, "radiomics", "results")

OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "radiomics_features.csv"
)


# ============================================================
# SETTINGS
# ============================================================

CLASSES = [
    "glioma",
    "meningioma",
    "notumor",
    "pituitary"
]


# ============================================================
# CREATE OUTPUT FOLDER
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# PYRADIOMICS CONFIGURATION
# ============================================================

params = {
    "binWidth": 25,
    "normalize": False,
    "force2D": True,
    "force2Ddimension": 0
}

extractor = featureextractor.RadiomicsFeatureExtractor(**params)

extractor.disableAllFeatures()

extractor.enableFeatureClassByName("firstorder")
extractor.enableFeatureClassByName("glcm")
extractor.enableFeatureClassByName("glrlm")
extractor.enableFeatureClassByName("glszm")
extractor.enableFeatureClassByName("gldm")
extractor.enableFeatureClassByName("ngtdm")
extractor.enableFeatureClassByName("shape")


# ============================================================
# CREATE AUTOMATIC FOREGROUND MASK
# ============================================================

def create_mask(image):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Remove very dark background
    _, mask = cv2.threshold(
        gray,
        10,
        255,
        cv2.THRESH_BINARY
    )

    # Remove small noise
    kernel = np.ones((3, 3), np.uint8)

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    # Make sure mask is binary
    mask = (mask > 0).astype(np.uint8)

    return mask


# ============================================================
# EXTRACT FEATURES
# ============================================================

all_features = []

total_images = 0
successful = 0
failed = 0


print()
print("=" * 60)
print("      RADIOMICS FEATURE EXTRACTION")
print("=" * 60)

print()
print("Dataset:")
print(DATASET_DIR)

print()
print("Output:")
print(OUTPUT_FILE)

print()


for class_name in CLASSES:

    class_dir = os.path.join(DATASET_DIR, class_name)

    if not os.path.exists(class_dir):

        print("WARNING: Folder not found:", class_dir)

        continue

    print()
    print("-" * 60)
    print("Processing:", class_name)
    print("-" * 60)

    image_files = [
        f for f in os.listdir(class_dir)
        if f.lower().endswith(
            (".jpg", ".jpeg", ".png", ".bmp")
        )
    ]

    print("Images found:", len(image_files))

    for index, filename in enumerate(image_files, start=1):

        image_path = os.path.join(
            class_dir,
            filename
        )

        total_images += 1

        try:

            # ------------------------------------------------
            # READ IMAGE
            # ------------------------------------------------

            image = cv2.imread(image_path)

            if image is None:

                print(
                    "Skipping unreadable image:",
                    filename
                )

                failed += 1

                continue


            # ------------------------------------------------
            # CONVERT TO GRAYSCALE
            # ------------------------------------------------

            gray = cv2.cvtColor(
                image,
                cv2.COLOR_BGR2GRAY
            )


            # ------------------------------------------------
            # CREATE AUTOMATIC MASK
            # ------------------------------------------------

            mask = create_mask(image)


            # ------------------------------------------------
            # CHECK MASK
            # ------------------------------------------------

            if np.sum(mask) == 0:

                print(
                    "Skipping empty mask:",
                    filename
                )

                failed += 1

                continue


            # ------------------------------------------------
            # SIMPLEITK IMAGE
            # ------------------------------------------------

            sitk_image = sitk.GetImageFromArray(
                gray.astype(np.float32)
            )


            # ------------------------------------------------
            # SIMPLEITK MASK
            # ------------------------------------------------

            sitk_mask = sitk.GetImageFromArray(
                mask.astype(np.uint8)
            )


            # ------------------------------------------------
            # EXTRACT RADIOMICS
            # ------------------------------------------------

            result = extractor.execute(
                sitk_image,
                sitk_mask
            )


            # ------------------------------------------------
            # SAVE FEATURES
            # ------------------------------------------------

            row = {
                "image": filename,
                "class": class_name
            }


            for key, value in result.items():

                # Remove diagnostic information
                if not key.startswith("diagnostics"):

                    try:

                        row[key] = float(value)

                    except (TypeError, ValueError):

                        pass


            all_features.append(row)

            successful += 1


            # ------------------------------------------------
            # PROGRESS
            # ------------------------------------------------

            if index % 100 == 0 or index == len(image_files):

                print(
                    f"Processed {index}/{len(image_files)}"
                )


        except Exception as e:

            failed += 1

            print(
                f"ERROR: {filename} -> {str(e)}"
            )


# ============================================================
# SAVE CSV
# ============================================================

print()
print("=" * 60)
print("Saving radiomics features...")
print("=" * 60)


if len(all_features) == 0:

    print()
    print("ERROR: No radiomics features were extracted.")
    print()
    print("Check the dataset path and image files.")
    print()

    raise SystemExit(1)


df = pd.DataFrame(all_features)

df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 60)
print("RADIOMICS EXTRACTION COMPLETED")
print("=" * 60)

print()

print("Total images :", total_images)
print("Successful   :", successful)
print("Failed       :", failed)

print()

print("Feature matrix shape:")
print(df.shape)

print()

print("Classes:")
print(df["class"].value_counts())

print()

print("Saved file:")
print(OUTPUT_FILE)

print()

print("=" * 60)