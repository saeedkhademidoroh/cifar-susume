# Standard library imports
import os
from pathlib import Path

# Third-party imports
from keras.api.callbacks import ModelCheckpoint

# Project-specific imports
from config import CONFIG
from log import log_to_json

def train_model(train_data, train_labels, model, model_name="mobilenet", verbose=1, result_file_path=None):
    """
    Trains a model with training data and logs the best checkpoint.

    Parameters:
        train_data (np.ndarray): Training inputs.
        train_labels (np.ndarray): One-hot encoded training labels.
        model (tf.keras.Model): Compiled Keras model.
        model_name (str): Name used for saved file/logs.
        verbose (int): Verbosity level (0=silent, 1=progress bar).
        result_file_path (Path): Path to the JSON file where results will be logged.

    Returns:
        tuple: (trained model, training history)
    """

    print("\n🎯 Train Model 🎯")

    os.makedirs(CONFIG.CHECKPOINT_PATH, exist_ok=True)
    checkpoint_file = CONFIG.CHECKPOINT_PATH / f"{model_name}_best.h5"

    checkpoint_cb = ModelCheckpoint(
        filepath=checkpoint_file,
        monitor='val_accuracy',
        save_best_only=True,
        save_weights_only=False,
        verbose=verbose
    )

    train_data = train_data[:5000]
    train_labels = train_labels[:5000]

    history = model.fit(
        x=train_data,
        y=train_labels,
        epochs=CONFIG.EPOCHS_COUNT,
        batch_size=CONFIG.BATCH_SIZE,
        validation_split=CONFIG.VALIDATION_SPLIT,
        callbacks=[checkpoint_cb],
        verbose=verbose,
    )

    best_val_acc = max(history.history.get("val_accuracy", [0]))

    if result_file_path:
        log_to_json(file_path=result_file_path, key="checkpoints", record={
            "model": model_name,
            "path": str(checkpoint_file),
            "val_accuracy": round(float(best_val_acc), 4),
            "epochs": len(history.history["loss"]),
            "status": "best"
        })

    return model, history

print("\n✅ train.py successfully executed")
