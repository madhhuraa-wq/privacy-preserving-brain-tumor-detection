import flwr as fl
import tensorflow as tf
import numpy as np
import os
import sys

# ============================================================
# SETTINGS
# ============================================================

IMG_SIZE = 64
NUM_CLASSES = 4
BATCH_SIZE = 32
EPOCHS = 1

# ============================================================
# DATASET PATH
# ============================================================

PROJECT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TRAINING_PATH = os.path.join(
    PROJECT_PATH,
    "dataset",
    "Training"
)

CLASS_NAMES = [
    "glioma",
    "meningioma",
    "notumor",
    "pituitary"
]

# ============================================================
# MODEL
# IMPORTANT: MUST MATCH SERVER EXACTLY
# ============================================================

def create_model():

    model = tf.keras.Sequential([

        tf.keras.layers.Input(
            shape=(64, 64, 3)
        ),

        tf.keras.layers.Conv2D(
            32,
            (3, 3),
            activation="relu"
        ),

        tf.keras.layers.MaxPooling2D(),

        tf.keras.layers.Conv2D(
            64,
            (3, 3),
            activation="relu"
        ),

        tf.keras.layers.MaxPooling2D(),

        tf.keras.layers.Conv2D(
            128,
            (3, 3),
            activation="relu"
        ),

        tf.keras.layers.MaxPooling2D(),

        tf.keras.layers.Flatten(),

        tf.keras.layers.Dense(
            128,
            activation="relu"
        ),

        tf.keras.layers.Dropout(
            0.5
        ),

        tf.keras.layers.Dense(
            4,
            activation="softmax"
        )
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


# ============================================================
# LOAD DATA FOR EACH FEDERATED CLIENT
# ============================================================

def load_client_data(client_id):

    print("\n========================================")
    print(f"Loading Client {client_id}")
    print("========================================")

    if not os.path.exists(TRAINING_PATH):

        raise FileNotFoundError(
            f"Training dataset not found:\n{TRAINING_PATH}"
        )

    images = []
    labels = []

    print(f"Dataset: {TRAINING_PATH}")
    print(f"Classes: {CLASS_NAMES}")

    # --------------------------------------------------------
    # LOAD ALL TRAINING IMAGES
    # --------------------------------------------------------

    for label, class_name in enumerate(CLASS_NAMES):

        class_path = os.path.join(
            TRAINING_PATH,
            class_name
        )

        print(f"Scanning {class_name}...")

        if not os.path.exists(class_path):

            print(
                f"WARNING: {class_path} not found"
            )

            continue

        for filename in os.listdir(class_path):

            filepath = os.path.join(
                class_path,
                filename
            )

            if not os.path.isfile(filepath):
                continue

            try:

                image = tf.keras.utils.load_img(
                    filepath,
                    target_size=(
                        IMG_SIZE,
                        IMG_SIZE
                    ),
                    color_mode="rgb"
                )

                image = tf.keras.utils.img_to_array(
                    image
                )

                image = image / 255.0

                images.append(image)
                labels.append(label)

            except Exception:

                print(
                    f"Skipping damaged image: {filename}"
                )

    if len(images) == 0:

        raise ValueError(
            "No training images were found."
        )

    X = np.array(
        images,
        dtype=np.float32
    )

    y = np.array(
        labels,
        dtype=np.int32
    )

    # --------------------------------------------------------
    # SHUFFLE DATA
    # --------------------------------------------------------

    np.random.seed(42)

    indices = np.random.permutation(
        len(X)
    )

    X = X[indices]
    y = y[indices]

    # --------------------------------------------------------
    # SPLIT DATA BETWEEN 4 FEDERATED CLIENTS
    # --------------------------------------------------------

    client_data = np.array_split(
        np.arange(len(X)),
        4
    )

    selected_indices = client_data[client_id]

    X_client = X[selected_indices]
    y_client = y[selected_indices]

    print(
        f"\nClient {client_id} samples: "
        f"{len(X_client)}"
    )

    print(
        f"Client {client_id} data loaded successfully."
    )

    return X_client, y_client


# ============================================================
# FEDERATED CLIENT
# ============================================================

class BrainTumorClient(
    fl.client.NumPyClient
):

    def __init__(self, client_id):

        self.client_id = client_id

        self.X, self.y = load_client_data(
            client_id
        )

        self.model = create_model()


    # ========================================================
    # GET PARAMETERS
    # ========================================================

    def get_parameters(self, config):

        return self.model.get_weights()


    # ========================================================
    # TRAIN
    # ========================================================

    def fit(
        self,
        parameters,
        config
    ):

        print(
            f"\nClient {self.client_id}: "
            f"Training started..."
        )

        # Receive global model
        self.model.set_weights(
            parameters
        )

        history = self.model.fit(

            self.X,

            self.y,

            epochs=EPOCHS,

            batch_size=BATCH_SIZE,

            verbose=1,

            shuffle=True
        )

        loss = history.history[
            "loss"
        ][-1]

        accuracy = history.history[
            "accuracy"
        ][-1]

        print(
            f"\nClient {self.client_id}: "
            f"Training complete"
        )

        print(
            f"Loss: {loss:.4f}"
        )

        print(
            f"Accuracy: {accuracy:.4f}"
        )

        return (

            self.model.get_weights(),

            len(self.X),

            {
                "accuracy":
                float(accuracy)
            }
        )


    # ========================================================
    # EVALUATE
    # ========================================================

    def evaluate(
        self,
        parameters,
        config
    ):

        self.model.set_weights(
            parameters
        )

        loss, accuracy = self.model.evaluate(

            self.X,

            self.y,

            verbose=0
        )

        print(
            f"\nClient {self.client_id}: "
            f"Evaluation accuracy = "
            f"{accuracy:.4f}"
        )

        return (

            float(loss),

            len(self.X),

            {
                "accuracy":
                float(accuracy)
            }
        )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    if len(sys.argv) < 2:

        print(
            "\nERROR: Please provide client ID."
        )

        print(
            "Use:"
        )

        print(
            "python federated_client.py 0"
        )

        sys.exit(1)

    client_id = int(
        sys.argv[1]
    )

    if client_id not in [0, 1, 2, 3]:

        print(
            "\nERROR: Client ID must be 0, 1, 2, or 3."
        )

        sys.exit(1)

    client = BrainTumorClient(
        client_id
    )

    print(
        f"\nConnecting Client {client_id} "
        f"to Flower server..."
    )

    fl.client.start_client(

        server_address="127.0.0.1:8080",

        client=client.to_client()
    )