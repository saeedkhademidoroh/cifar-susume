# Import third-party libraries
from keras.api.models import Model
from keras.api.layers import (
    Input, Conv2D, MaxPooling2D, Flatten, Dense,
    BatchNormalization, Activation, GlobalAveragePooling2D,
    DepthwiseConv2D, Add
)
from keras.api.optimizers import Adam, SGD
from keras.api.losses import SparseCategoricalCrossentropy


# Function to build a model
def build_model(model_number: int, config) -> Model:
    """
    Function to build and compile a model based on the given model_number.

    Supports several variants of VGG-style CNNs with optional BatchNorm,
    GlobalAveragePooling, separable convolutions, and residual connections.

    Args:
        model_number (int): Identifier for architecture variant (0–5)

    Returns:
        Model: A compiled Keras model instance
    """

    # Print header for function execution
    print("\n🎯  build_model\n")

    # Extract optimizer configuration from config
    optimizer_config = config.OPTIMIZER
    optimizer_type = optimizer_config["type"].lower()

    # Initialize optimizer according to config
    if optimizer_type == "sgd":
        optimizer = SGD(
            learning_rate=optimizer_config.get("learning_rate", 0.01),
            momentum=optimizer_config.get("momentum", 0.9)
        )
    elif optimizer_type == "adam":
        optimizer = Adam(
            learning_rate=optimizer_config.get("learning_rate", 0.001)
        )
    else:
        raise ValueError(f"❌ ValueError:\noptimizer_type={optimizer_type}\n")

    # Model 0: VGG-style with two Conv blocks and Dense(128), no BatchNorm
    if model_number == 0:
        input_layer = Input(shape=(32, 32, 3))

        # Block 1: Conv(32) x2 + MaxPool
        x = Conv2D(32, (3, 3), activation="relu", padding="same")(input_layer)
        x = Conv2D(32, (3, 3), activation="relu", padding="same")(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        # Block 2: Conv(64) x2 + MaxPool
        x = Conv2D(64, (3, 3), activation="relu", padding="same")(x)
        x = Conv2D(64, (3, 3), activation="relu", padding="same")(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        # Flatten + Dense(128)
        x = Flatten()(x)
        x = Dense(128, activation="relu")(x)
        prediction_layer = Dense(10, activation="softmax")(x)

    # Model 1: Model 0 + Batch Normalization
    elif model_number == 1:
        input_layer = Input(shape=(32, 32, 3))

        # Block 1: Conv(32) x2 + BN + ReLU + MaxPool
        x = Conv2D(32, (3, 3), padding="same")(input_layer)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = Conv2D(32, (3, 3), padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        # Block 2: Conv(64) x2 + BN + ReLU + MaxPool
        x = Conv2D(64, (3, 3), padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = Conv2D(64, (3, 3), padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        # Flatten + Dense(128)
        x = Flatten()(x)
        x = Dense(128, activation="relu")(x)
        prediction_layer = Dense(10, activation="softmax")(x)

    # Model 2: Model 1 with fewer filters, no Dense(128), use GAP
    elif model_number == 2:
        input_layer = Input(shape=(32, 32, 3))

        # Block 1: Conv(16) x2 + BN + ReLU + MaxPool
        x = Conv2D(16, (3, 3), padding="same")(input_layer)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = Conv2D(16, (3, 3), padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        # Block 2: Conv(32) x2 + BN + ReLU + MaxPool
        x = Conv2D(32, (3, 3), padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = Conv2D(32, (3, 3), padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        # GlobalAveragePooling instead of Flatten + Dense
        x = GlobalAveragePooling2D()(x)
        prediction_layer = Dense(10, activation="softmax")(x)

    # Model 3: Like model 1 but no Dense(128), keep original filters
    elif model_number == 3:
        input_layer = Input(shape=(32, 32, 3))

        # Block 1: Conv(32) x2 + BN + ReLU + MaxPool
        x = Conv2D(32, (3, 3), padding="same")(input_layer)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = Conv2D(32, (3, 3), padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        # Block 2: Conv(64) x2 + BN + ReLU + MaxPool
        x = Conv2D(64, (3, 3), padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = Conv2D(64, (3, 3), padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        x = GlobalAveragePooling2D()(x)
        prediction_layer = Dense(10, activation="softmax")(x)

    # Model 4: Depthwise separable convolution blocks
    elif model_number == 4:
        input_layer = Input(shape=(32, 32, 3))

        # Block 1: (Depthwise + Pointwise Conv) x2 + BN + ReLU + MaxPool
        x = DepthwiseConv2D((3, 3), padding="same")(input_layer)
        x = Conv2D(32, (1, 1), padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)

        x = DepthwiseConv2D((3, 3), padding="same")(x)
        x = Conv2D(32, (1, 1), padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        # Block 2: (Depthwise + Pointwise Conv) x2 + BN + ReLU + MaxPool
        x = DepthwiseConv2D((3, 3), padding="same")(x)
        x = Conv2D(64, (1, 1), padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)

        x = DepthwiseConv2D((3, 3), padding="same")(x)
        x = Conv2D(64, (1, 1), padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        x = GlobalAveragePooling2D()(x)
        prediction_layer = Dense(10, activation="softmax")(x)

    # Model 5: Residual connections (Add) + original filters + GAP
    elif model_number == 5:
        input_layer = Input(shape=(32, 32, 3))

        # Block 1: Conv(32) x2 + BN + Add shortcut + ReLU + MaxPool
        shortcut = Conv2D(32, (1, 1), padding="same")(input_layer)
        x = Conv2D(32, (3, 3), padding="same")(input_layer)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = Conv2D(32, (3, 3), padding="same")(x)
        x = BatchNormalization()(x)
        x = Add()([x, shortcut])
        x = Activation("relu")(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        # Block 2: Conv(64) x2 + BN + Add shortcut + ReLU + MaxPool
        shortcut = Conv2D(64, (1, 1), padding="same")(x)
        x = Conv2D(64, (3, 3), padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = Conv2D(64, (3, 3), padding="same")(x)
        x = BatchNormalization()(x)
        x = Add()([x, shortcut])
        x = Activation("relu")(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        x = GlobalAveragePooling2D()(x)
        prediction_layer = Dense(10, activation="softmax")(x)

    else:
        raise ValueError(f"❌ ValueError from model.py at build_model():\nmodel_number={model_number}\n")

    # Compile model with selected optimizer and loss/metrics
    model = Model(inputs=input_layer, outputs=prediction_layer)
    model.compile(optimizer=optimizer, loss=SparseCategoricalCrossentropy(), metrics=["accuracy"])

    # Print model architecture summary
    model.summary()

    return model  # Return compiled Keras model instance


# Print module successfully executed
print("\n✅  model.py successfully executed")
