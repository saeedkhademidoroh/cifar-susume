# Import third-party libraries
from keras.api.models import Model
from keras.api.layers import (
    Activation, Add, BatchNormalization, Conv2D,
    Dense, Dropout, GlobalAveragePooling2D, Input
)
from keras.api.optimizers import Adam, SGD
from keras.api.losses import SparseCategoricalCrossentropy


def _res_block(x, filters, regularizer, downsample=False):
    """
    Applies a 2-layer residual block with either identity or projection shortcut.

    Structure:
    - conv → BN → ReLU → conv → BN → Add(shortcut) → ReLU
    - Shortcut is identity if shape matches, else 1×1 conv with matching stride

    Args:
        x (Tensor): Input tensor.
        filters (int): Number of output channels.
        regularizer: Regularizer to apply (e.g. L2).
        downsample (bool): If True, apply stride=2 in the first conv layer.

    Returns:
        Tensor: Output of the residual block.
    """

    # Step 0: Print function execution header
    print("\n🎯  _res_block is executing ...")

    # Step 1: Determine stride (downsample or not)
    stride = 2 if downsample else 1

    # Step 2: Store the input tensor as shortcut for later addition
    shortcut = x

    # Step 3: First convolution layer with optional downsampling
    x = Conv2D(filters, (3, 3), strides=stride, padding="same", kernel_regularizer=regularizer)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    # Step 4: Second convolution layer (no downsampling)
    x = Conv2D(filters, (3, 3), padding="same", kernel_regularizer=regularizer)(x)
    x = BatchNormalization()(x)

    # Step 5: Apply 1x1 projection to shortcut if shape mismatch or downsampling
    if downsample or shortcut.shape[-1] != filters:
        shortcut = Conv2D(filters, (1, 1), strides=stride, padding="same", kernel_regularizer=regularizer)(shortcut)

    # Step 6: Add shortcut to the main path and apply ReLU
    x = Add()([x, shortcut])
    x = Activation("relu")(x)

    # Step 7: Return output tensor
    return x


def maybe_dropout(config, x):
    """
    Applies Dropout to a tensor if enabled in the configuration.

    Args:
        config (Config): Configuration object containing DROPOUT_MODE settings.
        x (Tensor): Input tensor to potentially apply Dropout to.

    Returns:
        Tensor: Either the original tensor or a Dropout-applied tensor.
    """

    # Step 0: Print function execution header
    print("\n🎯  maybe_dropout is executing ...")

    # Step 1: Extract dropout rate from config (default to 0.0 if missing)
    rate = config.DROPOUT_MODE.get("rate", 0.0)

    # Step 2: Validate that dropout rate is numeric
    if not isinstance(rate, (float, int)):
        raise ValueError(
            "\n\n❌  Error from model.py at maybe_dropout()!\n"
            "→ DROPOUT_MODE['rate'] must be a number (float or int)\n\n"
        )

    # Step 3: Apply Dropout if explicitly enabled in config
    if config.DROPOUT_MODE.get("enabled", False):
        return Dropout(rate)(x)

    # Step 4: Otherwise, return input tensor unchanged
    return x


def build_model(model_number: int, config) -> Model:
    """
    Builds and returns the specified model architecture.

    Supports multiple model variants identified by `model_number`. Each variant
    corresponds to a predefined model function (e.g., `build_model_9`).

    Args:
        model_number (int): Identifier for the model to be built.
        input_shape (tuple): Input shape of the images (e.g., (32, 32, 3)).
        num_classes (int): Number of output classes for classification.

    Returns:
        tf.keras.Model: Instantiated and compiled Keras model.

    Raises:
        ValueError: If `model_number` is not recognized.
    """

    # Print header for function execution
    print("\n🎯  build_model is executing ...")

    # Extract optimizer configuration block from the global config object
    optimizer_config = config.OPTIMIZER

    # Get optimizer type as lowercase string (e.g., 'adam', 'sgd')
    optimizer_type = str(optimizer_config.get("type", "adam")).lower()

    # Get base learning rate to be used by the optimizer
    learning_rate = optimizer_config.get("learning_rate", 0.001)

    # Get momentum value (used only if optimizer supports it, e.g., SGD)
    momentum = optimizer_config.get("momentum", 0.0)

    # Determine whether L2 regularization is enabled
    l2_enabled = config.L2_MODE.get("enabled", False)
    l2_mode = config.L2_MODE.get("mode", "").lower()

    if l2_enabled:
        # For other modes (e.g., 'layer'), logic can be extended here in future
        if l2_mode != "optimizer":
            raise ValueError("\n\n❌  ValueError from model.py at build_model()!\nUnsupported L2_MODE — only 'optimizer' is allowed for ResNet-style training\n\n")

        weight_decay = config.L2_MODE.get("lambda", 0.0)
        regularizer = None  # Avoid layer-level L2 to prevent double-counting
    else:
        weight_decay = 0.0
        regularizer = None  # No regularization at all

    # Initialize optimizer with or without weight decay
    if optimizer_type == "adam":
        # Use Adam optimizer with optional weight decay
        optimizer = Adam(learning_rate=learning_rate, weight_decay=weight_decay)

    elif optimizer_type == "sgd":
        # Use SGD optimizer with momentum and optional weight decay
        optimizer = SGD(learning_rate=learning_rate, momentum=momentum, weight_decay=weight_decay)

    else:
        # If weight decay is requested but the optimizer doesn't support it, raise an error
        if weight_decay > 0.0:
            raise ValueError(f"\n\n❌  ValueError from model.py at build_model()!\nOptimizer '{optimizer_type}' does not support weight_decay\n\n")

        # Dynamically resolve the optimizer class from keras.api.optimizers (e.g., RMSprop)
        optimizer_class = getattr(__import__("keras.api.optimizers"), optimizer_type.upper(), None)

        # Raise error if the optimizer type is not recognized
        if optimizer_class is None:
            raise ValueError(f"\n\n❌  ValueError from model.py at build_model()!\nUnknown optimizer type: '{optimizer_type}'\n\n")

        # Initialize the resolved optimizer class with learning rate (no weight decay support)
        optimizer = optimizer_class(learning_rate=learning_rate)

    # Disable all layer-level L2 regularization if optimizer-level is used
    if config.L2_MODE.get("enabled", False):
        if config.L2_MODE.get("mode") == "optimizer":
            regularizer = None  # Avoid duplicate penalty — rely solely on optimizer weight decay
        else:
            # Block unsupported regularization modes for consistency
            raise ValueError("\n\n❌  ValueError from model.py at build_model()!\nUnsupported L2_MODE — only 'optimizer' is allowed for ResNet-style training\n\n")
    else:
        # No L2 at all if disabled
        regularizer = None

    # Define input layer (CIFAR shape)
    input_layer = Input(shape=(32, 32, 3))

    # Validate supported model number
    if model_number not in [9]:
        raise ValueError(f"\n\n❌  ValueError from model.py at build_model()!\nUnsupported model_number={model_number}\n\n")

    # Model 9 — ResNet-20-style model with 3×3 conv blocks
    if model_number == 9:
        # Initial Conv block
        x = Conv2D(16, (3, 3), padding="same", kernel_regularizer=regularizer)(input_layer)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)

        # Stage 1 — 3 blocks, 16 filters
        for _ in range(3):
            x = _res_block(x, filters=16, regularizer=regularizer, downsample=False)

        # Stage 2 — downsample to 32, then 2 more blocks
        x = _res_block(x, filters=32, regularizer=regularizer, downsample=True)
        for _ in range(2):
            x = _res_block(x, filters=32, regularizer=regularizer, downsample=False)

        # Stage 3 — downsample to 64, then 2 more blocks
        x = _res_block(x, filters=64, regularizer=regularizer, downsample=True)
        for _ in range(2):
            x = _res_block(x, filters=64, regularizer=regularizer, downsample=False)

        # Final GAP → Dropout → Dense
        x = GlobalAveragePooling2D()(x)
        x = maybe_dropout(config, x)

        # Apply kernel_regularizer only if configured (i.e., not using optimizer-based weight decay)
        if regularizer:
            prediction_layer = Dense(10, activation="softmax", kernel_regularizer=regularizer)(x)
        else:
            prediction_layer = Dense(10, activation="softmax")(x)

    # Create the model instance from input and output tensors
    model = Model(inputs=input_layer, outputs=prediction_layer)

    # Print optimizer configuration for verification (e.g., learning rate, weight_decay)
    print(f"\n🔧  Optimizer is being configured for this training sesssion ...\n→ {optimizer}\n")

    # Compile the model with optimizer, loss function, and evaluation metric
    model.compile(
        optimizer=optimizer,
        loss=SparseCategoricalCrossentropy(),
        metrics=["accuracy"]
    )

    # Show model architecture
    model.summary()

    # Tag model with tracking metadata
    model.model_id = model_number  # Store the model version number for identification
    model.run_id = None            # Placeholder for run ID, to be set externally

    # Return the compiled Keras model instance
    return model


# Print module successfully executed
print("\n✅  model.py successfully executed.")
