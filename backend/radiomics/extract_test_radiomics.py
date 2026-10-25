import os
import cv2
import numpy as np
import pandas as pd
import SimpleITK as sitk
from radiomics import featureextractor

# ============================================================
# TEST RADIOMICS FEATURE EXTRACTION
# ============================================================

PROJECT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

DATASET_DIR = os.path.join(
    PROJECT_DIR,
    "dataset",
    "Testing"
)

OUTPUT_DIR = os.path.join(
    PROJECT_DIR,
    "radiomics",
    "results"
)

OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "radiomics_testing_features.csv"
)

CLASSES = [
    "glioma",
    "meningioma",
    "notumor",
    "pituitary"
]

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

# ============================================================
# SAME SETTINGS USED FOR TRAINING RADIOMICS
# ============================================================

params = {
    "binWidth": 25,
    "normalize": False,
    "force2D": True,
    "force2Ddimension": 0
}

extractor = featureextractor.RadiomicsFeatureExtractor(
    **params
)

extractor.disableAllFeatures()

extractor.enableFeatureClassByName("firstorder")
extractor.enableFeatureClassByName("glcm")
extractor.enableFeatureClassByName("glrlm")
extractor.enableFeatureClassByName("glszm")
extractor.enableFeatureClassByName("gldm")
extractor.enableFeatureClassByName("ngtdm")
extractor.enableFeatureClassByName("shape")

# ============================================================
# CREATE MASK
# ============================================================

def create_mask(image):

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    _, mask = cv2.threshold(
        gray,
        10,
        255,
        cv2.THRESH_BINARY
    )

    kernel = np.ones(
        (3, 3),
        np.uint8
    )

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

    # Convert to binary
    mask = (mask > 0).astype(np.uint8)

    # --------------------------------------------------------
    # If the entire image is selected, remove the outer border
    # so PyRadiomics has a valid ROI instead of a full-image ROI.
    # --------------------------------------------------------

    if np.mean(mask) > 0.995:

        mask[:2, :] = 0
        mask[-2:, :] = 0
        mask[:, :2] = 0
        mask[:, -2:] = 0

    # PyRadiomics label:
    # background = 0
    # ROI = 1
    mask = mask.astype(np.uint32)

    return mask


# ============================================================
# EXTRACT FEATURES
# ============================================================

all_features = []

total_images = 0
successful = 0
failed = 0

print("=" * 60)
print("TEST RADIOMICS FEATURE EXTRACTION")
print("=" * 60)

print("\nDataset:")
print(DATASET_DIR)

print("\nOutput:")
print(OUTPUT_FILE)

# ============================================================
# PROCESS CLASSES
# ============================================================

for class_name in CLASSES:

    class_dir = os.path.join(
        DATASET_DIR,
        class_name
    )

    print("\n" + "-" * 60)
    print("Processing:", class_name)
    print("-" * 60)

    if not os.path.exists(class_dir):

        print(
            "ERROR: Class folder not found:",
            class_dir
        )

        continue

    image_files = [
        f
        for f in os.listdir(class_dir)
        if f.lower().endswith(
            (".jpg", ".jpeg", ".png", ".bmp")
        )
    ]

    print(
        "Images found:",
        len(image_files)
    )

    for index, filename in enumerate(
        image_files,
        start=1
    ):

        total_images += 1

        image_path = os.path.join(
            class_dir,
            filename
        )

        try:

            # ------------------------------------------------
            # Read image
            # ------------------------------------------------

            image = cv2.imread(
                image_path
            )

            if image is None:

                print(
                    f"ERROR: Could not read {filename}"
                )

                failed += 1
                continue

            # ------------------------------------------------
            # Convert to grayscale
            # ------------------------------------------------

            gray = cv2.cvtColor(
                image,
                cv2.COLOR_BGR2GRAY
            )

            # ------------------------------------------------
            # Create mask
            # ------------------------------------------------

            mask = create_mask(
                image
            )

            if np.sum(mask) == 0:

                print(
                    f"ERROR: Empty mask -> {filename}"
                )

                failed += 1
                continue

            # ------------------------------------------------
            # SimpleITK image
            # ------------------------------------------------

            sitk_image = sitk.GetImageFromArray(
                gray.astype(np.float32)
            )

            # ------------------------------------------------
            # SimpleITK mask
            # ------------------------------------------------

            sitk_mask = sitk.GetImageFromArray(
                mask.astype(np.uint32)
            )

            # ------------------------------------------------
            # Match image and mask information
            # ------------------------------------------------

            sitk_mask.CopyInformation(
                sitk_image
            )

            # ------------------------------------------------
            # IMPORTANT:
            # Tell PyRadiomics that ROI label = 1
            # ------------------------------------------------

            result = extractor.execute(
                sitk_image,
                sitk_mask,
                label=1
            )

            # ------------------------------------------------
            # Save row
            # ------------------------------------------------

            row = {
                "image": filename,
                "class": class_name
            }

            for key, value in result.items():

                if not key.startswith(
                    "diagnostics"
                ):

                    try:

                        row[key] = float(
                            value
                        )

                    except:

                        pass

            all_features.append(
                row
            )

            successful += 1

            # ------------------------------------------------
            # Progress
            # ------------------------------------------------

            if (
                index % 100 == 0
                or index == len(image_files)
            ):

                print(
                    f"Processed "
                    f"{index}/{len(image_files)}"
                )

        except Exception as e:

            failed += 1

            print(
                f"ERROR: {filename} -> {e}"
            )


# ============================================================
# SAVE CSV
# ============================================================

print("\n" + "=" * 60)
print("Saving TEST radiomics features...")
print("=" * 60)

if len(all_features) == 0:

    print(
        "ERROR: No radiomics features extracted."
    )

    raise SystemExit(1)


df = pd.DataFrame(
    all_features
)

df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("TEST RADIOMICS EXTRACTION COMPLETED")
print("=" * 60)

print(
    "\nTotal images :",
    total_images
)

print(
    "Successful   :",
    successful
)

print(
    "Failed       :",
    failed
)

print(
    "\nFeature matrix shape:"
)

print(
    df.shape
)

print(
    "\nClasses:"
)

print(
    df["class"].value_counts()
)

print(
    "\nSaved file:"
)

print(
    OUTPUT_FILE
)

print("\n" + "=" * 60)
