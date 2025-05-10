# Import standard libraries
from timeit import default_timer as timer

# Third-party imports
from keras.api.layers import Input, Dense, GlobalAveragePooling2D
from keras.api.models import Model
from keras.api.optimizers import Adam
from keras.api.losses import CategoricalCrossentropy
from keras.api.applications import MobileNet
import numpy as np

# Function to create model
def build_model(model_number: int) -> Model:
    """
    Returns compiled model based on specified model number.

    Parameters:
    - model_number (int): Model variant to create (1 to N).

    Returns:
    - Compiled model and description (if any).
    """

    print("\n🎯 Build Model 🎯\n")

    if model_number == 1:
        input_layer = Input(shape=(32, 32, 3))
        base_model = MobileNet(input_tensor=input_layer, include_top=False, weights="imagenet")
        base_model.trainable = False

        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(256, activation="relu")(x)
        prediction_layer = Dense(10, activation="softmax")(x)

        model = Model(inputs=input_layer, outputs=prediction_layer, name="mobilenet_cifar")
        model.compile(optimizer=Adam(), loss=CategoricalCrossentropy(), metrics=["categorical_accuracy"])
        description = "MobileNet (frozen) + dense head for CIFAR-10"

        model.summary()
        start_time = timer()
        _ = model(np.random.rand(1, 32, 32, 3))
        elapsed_time = timer() - start_time
        print("\n🔹 Inference Time:\n")
        print(f"{elapsed_time:.6f} seconds (m{model_number})")

        return model, description

    elif model_number == 2:
        raise NotImplementedError("Model 2 is not implemented yet.")

    else:
        raise ValueError(f"❌ Invalid model number: {model_number}")

# Print confirmation message
print("\n✅ model.py successfully executed")
