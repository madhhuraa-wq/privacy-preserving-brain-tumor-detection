from flask import Flask, request, jsonify
import os
import pandas as pd
import joblib
import numpy as np

# ============================================================
# BRAIN TUMOR DETECTION - RADIOMICS FRONTEND
# ============================================================

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
# LOAD MODEL
# ============================================================

print("=" * 60)
print("BRAIN TUMOR DETECTION SYSTEM")
print("=" * 60)

print("\nLoading radiomics model...")

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
pca = joblib.load(PCA_PATH)

print("Random Forest loaded")
print("Scaler loaded")
print("PCA loaded")

# ============================================================
# LOAD TESTING RADIOMICS
# ============================================================

TEST_FILE = os.path.join(
    PROJECT_DIR,
    "radiomics",
    "results",
    "radiomics_testing_features.csv"
)

test_df = pd.read_csv(TEST_FILE)

FEATURE_COLUMNS = [
    column
    for column in test_df.columns
    if column not in ["image", "class"]
]

print(
    "Radiomics features:",
    len(FEATURE_COLUMNS)
)

print("=" * 60)


# ============================================================
# HOME PAGE
# ============================================================

@app.route("/")
def home():

    return """
<!DOCTYPE html>

<html>

<head>

<title>Brain Tumor Detection</title>

<style>

body {
    margin: 0;
    font-family: Arial, sans-serif;
    background: #eef2f7;
}

.header {
    background: #172554;
    color: white;
    padding: 25px;
    text-align: center;
}

.container {
    width: 650px;
    max-width: 90%;
    margin: 50px auto;
    background: white;
    padding: 35px;
    border-radius: 15px;
    box-shadow: 0px 5px 20px rgba(0,0,0,0.12);
}

h1 {
    margin-bottom: 5px;
}

.subtitle {
    color: #64748b;
}

.upload {
    margin-top: 30px;
    padding: 30px;
    border: 2px dashed #94a3b8;
    border-radius: 12px;
    text-align: center;
}

select,
button {
    margin-top: 20px;
    padding: 12px;
    font-size: 16px;
    border-radius: 8px;
}

select {
    width: 100%;
    border: 1px solid #cbd5e1;
}

button {
    width: 100%;
    border: none;
    background: #2563eb;
    color: white;
    cursor: pointer;
}

button:hover {
    background: #1d4ed8;
}

#result {
    margin-top: 30px;
    padding: 20px;
    border-radius: 10px;
    background: #f1f5f9;
    display: none;
}

.prediction {
    font-size: 25px;
    font-weight: bold;
    color: #172554;
}

.confidence {
    margin-top: 10px;
    font-size: 18px;
}

.info {
    margin-top: 30px;
    font-size: 14px;
    color: #64748b;
    line-height: 1.6;
}

</style>

</head>

<body>

<div class="header">

<h1>Brain Tumor Detection</h1>

<div>
Radiomics-Based MRI Classification
</div>

</div>


<div class="container">

<h2>Upload MRI</h2>

<p class="subtitle">
Select a radiomics feature record from the testing dataset.
</p>

<div class="upload">

<select id="imageSelect">

<option value="">
-- Select MRI Image --
</option>

IMAGE_OPTIONS

</select>

<button onclick="predict()">
Predict Tumor Type
</button>

</div>


<div id="result">

<div class="prediction" id="prediction"></div>

<div class="confidence" id="confidence"></div>

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

</div>


<script>

async function predict() {

    const image =
        document.getElementById("imageSelect").value;

    if (!image) {

        alert("Please select an MRI image.");

        return;
    }

    const response = await fetch(
        "/predict",
        {
            method: "POST",

            headers: {
                "Content-Type": "application/json"
            },

            body: JSON.stringify({
                image: image
            })
        }
    );

    const data = await response.json();

    if (data.error) {

        alert(data.error);

        return;
    }

    document.getElementById(
        "result"
    ).style.display = "block";

    document.getElementById(
        "prediction"
    ).innerHTML =
        "Prediction: " + data.prediction;

    document.getElementById(
        "confidence"
    ).innerHTML =
        "Confidence: " + data.confidence + "%";
}

</script>

</body>

</html>
"""


# ============================================================
# PREDICTION
# ============================================================

@app.route(
    "/predict",
    methods=["POST"]
)
def predict():

    try:

        data = request.get_json()

        image_name = data.get("image")

        if not image_name:

            return jsonify({
                "error": "No image selected."
            }), 400

        # ----------------------------------------------------
        # FIND IMAGE IN TEST RADIOMICS DATA
        # ----------------------------------------------------

        rows = test_df[
            test_df["image"] == image_name
        ]

        if len(rows) == 0:

            return jsonify({
                "error": "Image not found."
            }), 404

        row = rows.iloc[0]

        # ----------------------------------------------------
        # FEATURES
        # ----------------------------------------------------

        X = row[
            FEATURE_COLUMNS
        ].values.astype(
            np.float64
        ).reshape(
            1, -1
        )

        # ----------------------------------------------------
        # SCALING
        # ----------------------------------------------------

        X_scaled = scaler.transform(X)

        # ----------------------------------------------------
        # PCA
        # ----------------------------------------------------

        X_pca = pca.transform(
            X_scaled
        )

        # ----------------------------------------------------
        # PREDICTION
        # ----------------------------------------------------

        prediction = model.predict(
            X_pca
        )[0]

        probabilities = model.predict_proba(
            X_pca
        )[0]

        confidence = (
            np.max(probabilities) * 100
        )

        return jsonify({

            "prediction": str(
                prediction
            ),

            "confidence": round(
                float(confidence),
                2
            )

        })

    except Exception as e:

        print(
            "Prediction error:",
            str(e)
        )

        return jsonify({
            "error": str(e)
        }), 500


# ============================================================
# START SERVER
# ============================================================

if __name__ == "__main__":

    print("\nStarting Flask server...")

    print(
        "\nOpen in browser:"
    )

    print(
        "http://127.0.0.1:5000"
    )

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )