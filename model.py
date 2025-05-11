# Import standard libraries
from timeit import default_timer as timer

# Import third-party libraries
from keras.api.models import Model
from keras.api.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
from keras.api.optimizers import Adam
from keras.api.losses import SparseCategoricalCrossentropy


# Function to create model based on a model number
def build_model(model_number: int) -> Model:
    """
    Builds and compiles a model based on the specified model_number.

    Args:
        model_number (int): Identifier for model architecture.
            - 1: Simple CNN for sanity checks
            - 2: Compact VGG-style CNN for deeper training

    Returns:
        Model: A compiled Keras model ready for training.
    """
    print("\n🎯 build_model\n")

    if model_number == 1:

        input_layer = Input(shape=(32, 32, 3))

        x = Conv2D(32, (3, 3), activation="relu", padding="same")(input_layer)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Conv2D(64, (3, 3), activation="relu", padding="same")(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Flatten()(x)
        x = Dense(64, activation="relu")(x)

        prediction_layer = Dense(10, activation="softmax")(x)

        model = Model(inputs=input_layer, outputs=prediction_layer)
        model.compile(
            optimizer=Adam(),
            loss=SparseCategoricalCrossentropy(),
            metrics=["accuracy"]
        )

    elif model_number == 2:

        input_layer = Input(shape=(32, 32, 3))

        x = Conv2D(32, (3, 3), activation="relu", padding="same")(input_layer)
        x = Conv2D(32, (3, 3), activation="relu", padding="same")(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        x = Conv2D(64, (3, 3), activation="relu", padding="same")(x)
        x = Conv2D(64, (3, 3), activation="relu", padding="same")(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        x = Flatten()(x)
        x = Dense(128, activation="relu")(x)
        
        prediction_layer = Dense(10, activation="softmax")(x)

        model = Model(inputs=input_layer, outputs=prediction_layer)
        model.compile(
            optimizer=Adam(),
            loss=SparseCategoricalCrossentropy(),
            metrics=["accuracy"]
        )

    else:
        raise ValueError(f"❌ ValueError:\nmodel_number={model_number}\n")

    model.summary()

    return model


print("\n✅ model.py successfully executed")
