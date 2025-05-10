# Standard library imports
import os
from pathlib import Path

# Third-party imports
from keras.api.callbacks import ModelCheckpoint

# Project-specific imports
from config import CONFIG
from log import log_to_json


def train_model(train_data, train_labels, model, model_name="mobilenet", verbose=1):
    """
    Trains a model with training data and logs the best checkpoint.

    Parameters:
        train_data (np.ndarray): Training inputs.
        train_labels (np.ndarray): One-hot encoded training labels.
        model (tf.keras.Model): Compiled Keras model.
        model_name (str): Name used for saved file/logs.
        verbose (int): Verbosity level (0=silent, 1=progress bar).

    Returns:
        tuple: (trained model, training history)
    """

    print("\n🎯 Train Model 🎯")

    # Ensure checkpoint directory exists
    os.makedirs(CONFIG.CHECKPOINT_PATH, exist_ok=True)

    # Checkpoint file path
    checkpoint_file = CONFIG.CHECKPOINT_PATH / f"{model_name}_best.h5"

    # Keras model checkpoint callback
    checkpoint_cb = ModelCheckpoint(
        filepath=checkpoint_file,
        monitor='val_accuracy',
        save_best_only=CONFIG.SAVE_BEST_ONLY,
        save_weights_only=False,
        verbose=verbose
    )

    # Train the model
    history = model.fit(
        x=train_data,
        y=train_labels,
        epochs=CONFIG.EPOCHS,
        batch_size=CONFIG.BATCH_SIZE,
        validation_split=CONFIG.VALIDATION_SPLIT,
        callbacks=[checkpoint_cb],
        verbose=verbose,
    )

    # Log best checkpoint info
    best_val_acc = max(history.history.get("val_accuracy", [0]))
    log_to_json(key="checkpoints", record={
        "model": model_name,
        "path": str(checkpoint_file),
        "val_accuracy": round(float(best_val_acc), 4),
        "epochs": len(history.history["loss"]),
        "status": "best"
    })

    return model, history


# Print confirmation
print("\n✅ train.py successfully executed")
