import os
import cv2
import numpy as np
import SimpleITK as sitk
from radiomics import featureextractor

# ============================================================
# DIAGNOSE PYRADIOMICS EXTRACTION ERROR
# ============================================================

PROJECT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

DATASET_DIR = os.path.join(
    PROJECT_DIR,
    "dataset",
    "Testing"
)

TEST_IMAGES = [
    ("glioma", "Te-gl_143.jpg"),
    ("meningioma", "Te-me_138.jpg"),
    ("notumor", "Te-no_120.jpg")
]

# ============================================================
# SAME SETTINGS AS TRAINING
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

# Use shape2D because the pipeline uses force2D
extractor.enableFeatureClassByName("shape2D")


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

    return (mask > 0).astype(np.uint8)


# ============================================================
# TEST EXTRACTION
# ============================================================

print("=" * 60)
print("PYRADIOMICS ERROR DIAGNOSTIC")
print("=" * 60)

for class_name, filename in TEST_IMAGES:

    print("\n" + "-" * 60)
    print("Testing:", filename)
    print("Class:", class_name)
    print("-" * 60)

    image_path = os.path.join(
        DATASET_DIR,
        class_name,
        filename
    )

    print("Path:", image_path)

    image = cv2.imread(image_path)

    if image is None:
        print("ERROR: Could not read image")
        continue

    print("Image shape:", image.shape)

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    mask = create_mask(image)

    print("Mask pixels:", np.sum(mask))
    print("Mask percentage:", np.mean(mask) * 100, "%")

    sitk_image = sitk.GetImageFromArray(
        gray.astype(np.float32)
    )

    sitk_mask = sitk.GetImageFromArray(
        mask.astype(np.uint8)
    )

    print("\nRunning PyRadiomics...")

    try:

        result = extractor.execute(
            sitk_image,
            sitk_mask
        )

        feature_count = 0

        for key in result.keys():

            if not key.startswith("diagnostics"):
                feature_count += 1

        print("\nSUCCESS!")
        print(
            "Radiomics features extracted:",
            feature_count
        )

    except Exception as e:

        print("\nFAILED!")
        print("=" * 60)
        print("ERROR TYPE:")
        print(type(e).__name__)

        print("\nERROR MESSAGE:")
        print(str(e))

        print("\nFULL ERROR:")
        import traceback
        traceback.print_exc()

print("\n" + "=" * 60)
print("DIAGNOSTIC COMPLETED")
print("=" * 60)