import flwr as fl
import tensorflow as tf
import numpy as np
import os


print("\n==========================================")
print("     Brain Tumor Federated Learning")
print("==========================================")

NUM_ROUNDS = 4
MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "federated_global_model.keras"
)


# ============================================================
# MODEL
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
# CREATE INITIAL MODEL
# ============================================================

model = create_model()

initial_parameters = fl.common.ndarrays_to_parameters(
    model.get_weights()
)


# ============================================================
# CUSTOM FEDAVG
# ============================================================

class SaveModelFedAvg(fl.server.strategy.FedAvg):

    def __init__(self, **kwargs):

        super().__init__(**kwargs)

        self.final_parameters = None


    def aggregate_fit(
        self,
        server_round,
        results,
        failures
    ):

        aggregated_parameters, metrics = super().aggregate_fit(
            server_round,
            results,
            failures
        )

        if aggregated_parameters is not None:

            self.final_parameters = aggregated_parameters

            print(
                f"\nGlobal model updated after round "
                f"{server_round}"
            )

        return aggregated_parameters, metrics


# ============================================================
# FEDERATED STRATEGY
# ============================================================

strategy = SaveModelFedAvg(

    fraction_fit=1.0,

    fraction_evaluate=1.0,

    min_fit_clients=4,

    min_evaluate_clients=4,

    min_available_clients=4,

    initial_parameters=initial_parameters
)


# ============================================================
# START SERVER
# ============================================================

print(f"\nFederated rounds: {NUM_ROUNDS}")
print("Waiting for clients...\n")


history = fl.server.start_server(

    server_address="127.0.0.1:8080",

    config=fl.server.ServerConfig(
        num_rounds=NUM_ROUNDS
    ),

    strategy=strategy
)


# ============================================================
# SAVE FINAL GLOBAL MODEL
# ============================================================

print("\n==========================================")
print("Federated Learning Finished")
print("==========================================")

if strategy.final_parameters is None:

    print(
        "\nERROR: Final global parameters were not received."
    )

else:

    print("\nExtracting final global parameters...")

    final_weights = fl.common.parameters_to_ndarrays(
        strategy.final_parameters
    )

    final_model = create_model()

    final_model.set_weights(
        final_weights
    )

    final_model.save(
        MODEL_PATH
    )

    print("\nFinal global model saved successfully!")

    print(
        f"File: {MODEL_PATH}"
    )

    print("\n==========================================")
    print("Federated Learning Complete")
    print("==========================================")