# Import third-party libraries
from keras.api.models import Model
from keras.api.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Activation
from keras.api.optimizers import Adam
from keras.api.losses import SparseCategoricalCrossentropy


# Function to build a model based on model_number
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

    # Print header for function execution
    print("\n🎯 build_model\n")

    # Sanity check model
    if model_number == 0:

        # Input layer for 32x32 RGB images
        input_layer = Input(shape=(32, 32, 3))

        # First convolution + pooling
        x = Conv2D(32, (3, 3), activation="relu", padding="same")(input_layer)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        # Second convolution + pooling
        x = Conv2D(64, (3, 3), activation="relu", padding="same")(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        # Flatten and fully connected layers
        x = Flatten()(x)
        x = Dense(64, activation="relu")(x)

        # Output softmax layer for 10 CIFAR-10 classes
        prediction_layer = Dense(10, activation="softmax")(x)

        # Build and compile the model
        model = Model(inputs=input_layer, outputs=prediction_layer)
        model.compile(
            optimizer=Adam(),
            loss=SparseCategoricalCrossentropy(),
            metrics=["accuracy"]
        )

    # VGG style model
    elif model_number == 1:

        input_layer = Input(shape=(32, 32, 3))

        # Block 1: Two conv layers + pooling
        x = Conv2D(32, (3, 3), activation="relu", padding="same")(input_layer)
        x = Conv2D(32, (3, 3), activation="relu", padding="same")(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        # Block 2: Two conv layers + pooling
        x = Conv2D(64, (3, 3), activation="relu", padding="same")(x)
        x = Conv2D(64, (3, 3), activation="relu", padding="same")(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        # Fully connected layers
        x = Flatten()(x)
        x = Dense(128, activation="relu")(x)

        # Output layer
        prediction_layer = Dense(10, activation="softmax")(x)

        # Build and compile the model
        model = Model(inputs=input_layer, outputs=prediction_layer)
        model.compile(
            optimizer=Adam(),
            loss=SparseCategoricalCrossentropy(),
            metrics=["accuracy"]
        )

    # m2 = m1 + Batch Normalization
    elif model_number == 2:

        input_layer = Input(shape=(32, 32, 3))

        # Block 1: Two conv layers + BN + pooling
        x = Conv2D(32, (3, 3), padding="same")(input_layer)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = Conv2D(32, (3, 3), padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        # Block 2: Two conv layers + BN + pooling
        x = Conv2D(64, (3, 3), padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = Conv2D(64, (3, 3), padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        # Fully connected layers
        x = Flatten()(x)
        x = Dense(128, activation="relu")(x)

        # Output layer
        prediction_layer = Dense(10, activation="softmax")(x)

        # Build and compile the model
        model = Model(inputs=input_layer, outputs=prediction_layer)
        model.compile(
            optimizer=Adam(),
            loss=SparseCategoricalCrossentropy(),
            metrics=["accuracy"]
        )

    else:
        # Raise error if unsupported model_number is given
        raise ValueError(f"❌ ValueError:\nmodel_number={model_number}\n")

    # Print model architecture summary to console
    model.summary()

    # Return built model
    return model


print("\n✅ model.py successfully executed")
