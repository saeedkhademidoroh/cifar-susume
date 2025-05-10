# Import standard libraries
from timeit import default_timer as timer

# Third-party imports
from keras.api.layers import Input, Dense, GlobalAveragePooling2D
from keras.api.models import Model
from keras.api.optimizers import Adam
from keras.api.losses import CategoricalCrossentropy
from keras.api.applications import MobileNet
import numpy as np


def benchmark_inference(model, input_shape, model_number=None):
    """
    Measures and prints inference time for a single forward pass.

    Args:
        model (tf.keras.Model): The compiled model to benchmark.
        input_shape (tuple): Shape of the input (excluding batch size).
        model_number (int, optional): If provided, included in the print output.
    """
    sample_input = np.random.rand(1, *input_shape).astype(np.float32)
    start_time = timer()
    _ = model(sample_input)
    elapsed_time = timer() - start_time
    print("\n🔹 Inference Time:\n")
    label = f" (m{model_number})" if model_number is not None else ""
    print(f"{elapsed_time:.6f} seconds{label}")

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
        base_model = MobileNet(input_tensor=input_layer, include_top=False, weights=None)
        base_model.trainable = False

        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(256, activation="relu")(x)
        prediction_layer = Dense(10, activation="softmax")(x)

        model = Model(inputs=input_layer, outputs=prediction_layer, name="mobilenet_cifar")
        model.compile(
            optimizer=Adam(),
            loss=CategoricalCrossentropy(),
            metrics=["accuracy"]
        )
        description = "MobileNet (frozen) + dense head for CIFAR-10"

        model.summary()

        benchmark_inference(model, input_shape=(32, 32, 3), model_number=model_number)
        
        return model, description

    elif model_number == 2:
        raise NotImplementedError("Model 2 is not implemented yet.")

    else:
        raise ValueError(f"❌ Invalid model number: {model_number}")

# Print confirmation message
print("\n✅ model.py successfully executed")
