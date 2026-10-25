from flask import Flask, request, jsonify
import os
import subprocess
import json
import tempfile
import numpy as np
import joblib

app = Flask(__name__)

# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

MODEL_DIR = os.path.join(
    PROJECT_DIR,
    "models",
    "radiomics"
)

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "radiomics_random_forest.pkl"
)

SCALER_PATH = os.path.join(
    MODEL_DIR,
    "radiomics_scaler.pkl"
)

PCA_PATH = os.path.join(
    MODEL_DIR,
    "radiomics_pca.pkl"
)

# ============================================================
# RADIOMICS39 PYTHON
# ============================================================

RADIOMICS_PYTHON = (
    r"C:\Users\MADHURA M\anaconda3\envs"
    r"\radiomics39\python.exe"
)

EXTRACTOR_SCRIPT = os.path.join(
    PROJECT_DIR,
    "frontend",
    "extract_uploaded.py"
)

# ============================================================
# LOAD TRAINED MODEL
# ============================================================

print("=" * 60)
print("BRAIN TUMOR DETECTION")
print("RADIOMICS-BASED MRI CLASSIFICATION")
print("=" * 60)

print("\nLoading trained model...")

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
pca = joblib.load(PCA_PATH)

print("Random Forest: OK")
print("Scaler: OK")
print("PCA: OK")

# ============================================================
# EXPECTED FEATURES
# ============================================================

EXPECTED_FEATURES = list(
    scaler.feature_names_in_
)

print(
    "Expected radiomics features:",
    len(EXPECTED_FEATURES)
)

# ============================================================
# HOME PAGE
# ============================================================

@app.route("/")
def home():

    return """
<!DOCTYPE html>

<html>

<head>

<meta charset="UTF-8">

<meta name="viewport"
      content="width=device-width, initial-scale=1.0">

<title>Brain Tumor Detection</title>

<style>

* {
    box-sizing: border-box;
}

body {
    margin: 0;
    font-family: Arial, sans-serif;
    background: #eef2f7;
    color: #1e293b;
}

.header {
    background: #172554;
    color: white;
    padding: 30px 20px;
    text-align: center;
}

.header h1 {
    margin: 0 0 8px 0;
}

.header p {
    margin: 0;
    opacity: 0.9;
}

.container {
    width: 650px;
    max-width: 92%;
    margin: 45px auto;
    background: white;
    padding: 35px;
    border-radius: 16px;
    box-shadow: 0 5px 25px rgba(0,0,0,0.10);
}

h2 {
    color: #172554;
    margin-top: 0;
}

.description {
    color: #64748b;
    line-height: 1.6;
}

input[type="file"] {
    width: 100%;
    margin-top: 20px;
    padding: 14px;
    border: 1px solid #cbd5e1;
    border-radius: 8px;
    background: white;
}

#preview {
    width: 100%;
    max-height: 350px;
    object-fit: contain;
    margin-top: 20px;
    display: none;
    border-radius: 10px;
    border: 1px solid #e2e8f0;
}

button {
    width: 100%;
    padding: 14px;
    margin-top: 20px;
    border: none;
    border-radius: 8px;
    background: #2563eb;
    color: white;
    font-size: 16px;
    cursor: pointer;
}

button:hover {
    background: #1d4ed8;
}

button:disabled {
    background: #94a3b8;
    cursor: not-allowed;
}

#result {
    display: none;
    margin-top: 25px;
    padding: 22px;
    border-radius: 12px;
    background: #f1f5f9;
}

.prediction {
    font-size: 27px;
    font-weight: bold;
    color: #172554;
}

.confidence {
    margin-top: 10px;
    font-size: 18px;
}

.info {
    margin-top: 30px;
    padding-top: 20px;
    border-top: 1px solid #e2e8f0;
    color: #64748b;
    line-height: 1.8;
}

.warning {
    margin-top: 20px;
    font-size: 13px;
    color: #64748b;
    line-height: 1.5;
}

</style>

</head>

<body>

<div class="header">

<h1>Brain Tumor Detection</h1>

<p>Radiomics-Based MRI Classification</p>

</div>


<div class="container">

<h2>Upload MRI Image</h2>

<p class="description">
Upload an MRI image for radiomics-based tumor classification.
The uploaded image is processed by the radiomics pipeline.
</p>


<input
    type="file"
    id="image"
    accept=".jpg,.jpeg,.png,.bmp"
    onchange="previewImage()"
>


<img
    id="preview"
>


<button
    id="predictButton"
    onclick="predict()"
>
    Analyze MRI
</button>


<div id="result">

    <div
        class="prediction"
        id="prediction">
    </div>

    <div
        class="confidence"
        id="confidence">
    </div>

</div>


<div class="info">

<strong>Model:</strong>
Radiomics + PCA + Random Forest

<br>

<strong>Radiomics Features:</strong>
93

<br>

<strong>PCA Components:</strong>
12

<br>

<strong>Test Accuracy:</strong>
86.75%

</div>


<div class="warning">

Prototype for academic project demonstration.
This system is not a medical diagnostic device.

</div>

</div>


<script>

function previewImage() {

    const file =
        document.getElementById(
            "image"
        ).files[0];

    if (!file) {
        return;
    }

    const preview =
        document.getElementById(
            "preview"
        );

    preview.src =
        URL.createObjectURL(file);

    preview.style.display =
        "block";
}


async function predict() {

    const file =
        document.getElementById(
            "image"
        ).files[0];

    if (!file) {

        alert(
            "Please upload an MRI image."
        );

        return;
    }

    const button =
        document.getElementById(
            "predictButton"
        );

    const result =
        document.getElementById(
            "result"
        );

    const prediction =
        document.getElementById(
            "prediction"
        );

    const confidence =
        document.getElementById(
            "confidence"
        );


    button.disabled = true;

    button.innerText =
        "Analyzing MRI...";

    result.style.display =
        "block";

    prediction.innerText =
        "Processing...";

    confidence.innerText =
        "";


    const formData =
        new FormData();

    formData.append(
        "image",
        file
    );


    try {

        const response =
            await fetch(
                "/predict",
                {
                    method: "POST",
                    body: formData
                }
            );


        const data =
            await response.json();


        if (!response.ok ||
            data.error) {

            prediction.innerText =
                "Prediction Error";

            confidence.innerText =
                data.error ||
                "Unable to process the MRI.";

            return;
        }


        prediction.innerText =
            "Prediction: " +
            data.prediction;


        confidence.innerText =
            "Confidence: " +
            data.confidence +
            "%";

    }

    catch (error) {

        prediction.innerText =
            "Server Error";

        confidence.innerText =
            error.toString();

    }

    finally {

        button.disabled = false;

        button.innerText =
            "Analyze MRI";
    }
}

</script>

</body>

</html>
"""


# ============================================================
# PREDICTION API
# ============================================================

@app.route(
    "/predict",
    methods=["POST"]
)

def predict():

    temporary_file = None

    try:

        # ----------------------------------------------------
        # CHECK UPLOAD
        # ----------------------------------------------------

        if "image" not in request.files:

            return jsonify({
                "error":
                "No MRI image uploaded."
            }), 400


        uploaded_file = request.files["image"]


        if uploaded_file.filename == "":

            return jsonify({
                "error":
                "No MRI image selected."
            }), 400


        # ----------------------------------------------------
        # CHECK FILE TYPE
        # ----------------------------------------------------

        extension = os.path.splitext(
            uploaded_file.filename
        )[1].lower()


        allowed_extensions = [
            ".jpg",
            ".jpeg",
            ".png",
            ".bmp"
        ]


        if extension not in allowed_extensions:

            return jsonify({
                "error":
                "Please upload JPG, JPEG, PNG or BMP."
            }), 400


        # ----------------------------------------------------
        # SAVE TEMPORARY IMAGE
        # ----------------------------------------------------

        with tempfile.NamedTemporaryFile(
            suffix=extension,
            delete=False
        ) as temp_file:

            temporary_file = temp_file.name

            uploaded_file.save(
                temporary_file
            )


        print(
            "\nProcessing MRI:",
            uploaded_file.filename
        )


        # ----------------------------------------------------
        # RUN PYRADIOMICS USING RADIOMICS39
        # ----------------------------------------------------

        process = subprocess.run(
            [
                RADIOMICS_PYTHON,
                EXTRACTOR_SCRIPT,
                temporary_file
            ],
            capture_output=True,
            text=True,
            timeout=120
        )


        if process.returncode != 0:

            print(
                "Radiomics error:",
                process.stderr
            )

            return jsonify({
                "error":
                "Radiomics extraction failed:\n"
                + process.stderr[-2000:]
            }), 500


        # ----------------------------------------------------
        # READ JSON OUTPUT
        # ----------------------------------------------------

        output_lines = (
            process.stdout
            .strip()
            .splitlines()
        )


        if not output_lines:

            return jsonify({
                "error":
                "Radiomics extraction returned no features."
            }), 500


        features = json.loads(
            output_lines[-1]
        )


        print(
            "Extracted features:",
            len(features)
        )


        # ----------------------------------------------------
        # ENSURE REQUIRED FEATURES EXIST
        # ----------------------------------------------------

        missing_features = [
            feature
            for feature in EXPECTED_FEATURES
            if feature not in features
        ]


        if missing_features:

            return jsonify({
                "error":
                f"{len(missing_features)} required "
                "radiomics features are missing."
            }), 500


        # ----------------------------------------------------
        # CREATE FEATURE VECTOR
        # ----------------------------------------------------

        X = np.array(
            [
                features[feature]
                for feature in EXPECTED_FEATURES
            ],
            dtype=np.float64
        ).reshape(
            1,
            -1
        )


        print(
            "Feature matrix:",
            X.shape
        )


        # ----------------------------------------------------
        # SCALING
        # ----------------------------------------------------

        X_scaled = scaler.transform(
            X
        )


        # ----------------------------------------------------
        # PCA
        # ----------------------------------------------------

        X_pca = pca.transform(
            X_scaled
        )


        print(
            "PCA matrix:",
            X_pca.shape
        )


        # ----------------------------------------------------
        # RANDOM FOREST
        # ----------------------------------------------------

        prediction = model.predict(
            X_pca
        )[0]


        probabilities = (
            model.predict_proba(
                X_pca
            )[0]
        )


        confidence = (
            float(
                np.max(
                    probabilities
                )
            ) * 100
        )


        # ----------------------------------------------------
        # DISPLAY NAME
        # ----------------------------------------------------

        display_names = {

            "glioma":
                "Glioma",

            "meningioma":
                "Meningioma",

            "notumor":
                "No Tumor",

            "pituitary":
                "Pituitary Tumor"
        }


        result = display_names.get(
            str(prediction),
            str(prediction)
        )


        print(
            "Prediction:",
            result
        )


        print(
            "Confidence:",
            round(
                confidence,
                2
            ),
            "%"
        )


        return jsonify({

            "prediction":
                result,

            "confidence":
                round(
                    confidence,
                    2
                )
        })


    except subprocess.TimeoutExpired:

        return jsonify({
            "error":
            "Radiomics extraction timed out."
        }), 500


    except Exception as e:

        print(
            "Prediction error:",
            repr(e)
        )

        return jsonify({
            "error":
            str(e)
        }), 500


    finally:

        # ----------------------------------------------------
        # DELETE TEMPORARY IMAGE
        # ----------------------------------------------------

        if (
            temporary_file
            and
            os.path.exists(
                temporary_file
            )
        ):

            try:
                os.remove(
                    temporary_file
                )

            except Exception:
                pass


# ============================================================
# START SERVER
# ============================================================

if __name__ == "__main__":

    print("\n" + "=" * 60)

    print(
        "FLASK SERVER STARTING"
    )

    print("=" * 60)

    print(
        "\nOpen in your browser:"
    )

    print(
        "http://127.0.0.1:5000"
    )

    print(
        "\nPress CTRL+C to stop."
    )


    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False
    )