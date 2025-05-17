# Import third-party libraries
from keras.api.models import Model
from keras.api.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Activation, GlobalAveragePooling2D
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

    # m0 = Baseline VGG-style (no BatchNorm, uses Dense(128))
    if model_number == 0:

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

    # m1 = m0 + Batch Normalization
    elif model_number == 1:

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

    # m2 = m1 - Dense(128) + Reduced Filters + GlobalAveragePooling
    elif model_number == 2:

        input_layer = Input(shape=(32, 32, 3))

        # Block 1 – 2×Conv(16) + BN + ReLU + MaxPool
        x = Conv2D(16, (3, 3), padding="same")(input_layer)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = Conv2D(16, (3, 3), padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        # Block 2 – 2×Conv(32) + BN + ReLU + MaxPool
        x = Conv2D(32, (3, 3), padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = Conv2D(32, (3, 3), padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        # GAP – replaces Flatten + Dense(128) to reduce params
        x = GlobalAveragePooling2D()(x)

        # Output layer – Dense(10) softmax classifier
        prediction_layer = Dense(10, activation="softmax")(x)

        model = Model(inputs=input_layer, outputs=prediction_layer)
        model.compile(
            optimizer=Adam(),
            loss=SparseCategoricalCrossentropy(),
            metrics=["accuracy"]
        )

    # m3 = m1 - Dense(128) + GlobalAveragePooling (original filters)
    elif model_number == 3:

        input_layer = Input(shape=(32, 32, 3))

        # Block 1 – 2×Conv(32) + BN + ReLU + MaxPool
        x = Conv2D(32, (3, 3), padding="same")(input_layer)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = Conv2D(32, (3, 3), padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        # Block 2 – 2×Conv(64) + BN + ReLU + MaxPool
        x = Conv2D(64, (3, 3), padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = Conv2D(64, (3, 3), padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        # GAP – replaces Flatten + Dense(128) to reduce params
        x = GlobalAveragePooling2D()(x)

        # Output layer – Dense(10) softmax classifier
        prediction_layer = Dense(10, activation="softmax")(x)

        model = Model(inputs=input_layer, outputs=prediction_layer)
        model.compile(
            optimizer=Adam(),
            loss=SparseCategoricalCrossentropy(),
            metrics=["accuracy"]
        )

    # m4 = m3 + Depthwise Separable Convs
    elif model_number == 4:

        from keras.api.layers import DepthwiseConv2D

        input_layer = Input(shape=(32, 32, 3))

        # Block 1 – 2×(Depthwise + Pointwise) + BN + ReLU + MaxPool
        x = DepthwiseConv2D((3, 3), padding="same")(input_layer)
        x = Conv2D(32, (1, 1), padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)

        x = DepthwiseConv2D((3, 3), padding="same")(x)
        x = Conv2D(32, (1, 1), padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)

        x = MaxPooling2D(pool_size=(2, 2))(x)

        # Block 2 – 2×(Depthwise + Pointwise) + BN + ReLU + MaxPool
        x = DepthwiseConv2D((3, 3), padding="same")(x)
        x = Conv2D(64, (1, 1), padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)

        x = DepthwiseConv2D((3, 3), padding="same")(x)
        x = Conv2D(64, (1, 1), padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)

        x = MaxPooling2D(pool_size=(2, 2))(x)

        # GAP + Dense(10)
        x = GlobalAveragePooling2D()(x)
        prediction_layer = Dense(10, activation="softmax")(x)

        model = Model(inputs=input_layer, outputs=prediction_layer)
        model.compile(
            optimizer=Adam(),
            loss=SparseCategoricalCrossentropy(),
            metrics=["accuracy"]
        )

    # m5 = m3 + Residual Connections (1×1 skip) + ReLU after Add
    elif model_number == 5:

        from keras.api.layers import Add

        input_layer = Input(shape=(32, 32, 3))

        # Block 1 – Residual(Conv-BN-ReLU → Conv-BN) + Add + ReLU → MaxPool
        shortcut = Conv2D(32, (1, 1), padding="same")(input_layer)

        x = Conv2D(32, (3, 3), padding="same")(input_layer)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = Conv2D(32, (3, 3), padding="same")(x)
        x = BatchNormalization()(x)

        x = Add()([x, shortcut])
        x = Activation("relu")(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        # Block 2 – Residual(Conv-BN-ReLU → Conv-BN) + Add + ReLU → MaxPool
        shortcut = Conv2D(64, (1, 1), padding="same")(x)

        x = Conv2D(64, (3, 3), padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = Conv2D(64, (3, 3), padding="same")(x)
        x = BatchNormalization()(x)

        x = Add()([x, shortcut])
        x = Activation("relu")(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        # GAP + Classifier
        x = GlobalAveragePooling2D()(x)
        prediction_layer = Dense(10, activation="softmax")(x)

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