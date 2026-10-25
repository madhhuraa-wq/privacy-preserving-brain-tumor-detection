import sys
import os
import json
import cv2
import numpy as np
import SimpleITK as sitk
from radiomics import featureextractor

IMAGE_PATH = sys.argv[1]

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

    mask = (mask > 0).astype(np.uint8)

    if np.mean(mask) > 0.995:
        mask[:2, :] = 0
        mask[-2:, :] = 0
        mask[:, :2] = 0
        mask[:, -2:] = 0

    return mask.astype(np.uint32)


image = cv2.imread(IMAGE_PATH)

if image is None:
    raise ValueError("Could not read uploaded image.")

gray = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2GRAY
)

mask = create_mask(image)

if np.sum(mask) == 0:
    raise ValueError("Could not create a valid MRI mask.")

sitk_image = sitk.GetImageFromArray(
    gray.astype(np.float32)
)

sitk_mask = sitk.GetImageFromArray(
    mask.astype(np.uint32)
)

sitk_mask.CopyInformation(
    sitk_image
)

result = extractor.execute(
    sitk_image,
    sitk_mask,
    label=1
)

features = {}

for key, value in result.items():

    if key.startswith("diagnostics"):
        continue

    try:
        features[key] = float(value)
    except:
        pass

print(json.dumps(features))