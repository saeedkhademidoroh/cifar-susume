# Import standard libraries
import os
import json
from pathlib import Path

# Import third-party libraries
from keras.api.callbacks import Callback, ModelCheckpoint
from keras.api.models import load_model

# Import project-specific libraries
from config import CONFIG
from log import log_to_json


# Function to define a custom recovery callback that saves the model and state after each epoch
class RecoveryCheckpoint(Callback):
    """
    Custom Keras callback to save a recovery checkpoint after each epoch.
    Saves both the model and a state.json file with current epoch.
    """

    # Constructor to prepare checkpoint directory and file paths
    def __init__(self, ckpt_dir: Path):
        """
        Initializes recovery paths and ensures checkpoint folder exists.
        """

        # Print header for function execution
        print("\n🎯 RecoveryCheckpoint.__init__ 🎯")

        super().__init__()
        self.ckpt_dir = ckpt_dir
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.model_path = ckpt_dir / "latest.keras"
        self.state_path = ckpt_dir / "state.json"

    # Function to save model and state.json after each epoch
    def on_epoch_end(self, epoch, logs=None):
        """
        Saves model and training state at the end of each epoch.
        """

        # Print header for function execution
        print("\n🎯 RecoveryCheckpoint.on_epoch_end 🎯\n")

        self.model.save(self.model_path)
        with open(self.state_path, "w") as f:
            json.dump({"initial_epoch": epoch + 1}, f)
        print(f"💾 Saved recovery checkpoint (epoch {epoch + 1})")



# Function to prepare ModelCheckpoint callbacks
def get_checkpoint_callbacks(model_ckpt_dir: Path, verbose: int = 1):
    """
    Returns a list of Keras callbacks to save both best and per-epoch checkpoints.
    """

    # Print header for function execution
    print("\n🎯 get_checkpoint_callbacks 🎯")

    # Define file paths for best and epoch-based checkpoints
    best_model_path = model_ckpt_dir / "best.keras"
    per_epoch_path = model_ckpt_dir / "epoch_{epoch:02d}.keras"

    return [
        ModelCheckpoint(
            filepath=best_model_path,
            monitor='val_accuracy',
            save_best_only=True,
            save_weights_only=False,
            verbose=verbose
        ),
        ModelCheckpoint(
            filepath=per_epoch_path,
            save_best_only=False,
            save_weights_only=False,
            verbose=0
        )
    ]



# Function to resume training state from checkpoint if available
def load_training_state(model_ckpt_dir: Path):
    """
    Loads model and epoch state from previous checkpoint if available.

    Returns:
        tuple: (loaded model or None, starting epoch)
    """

    # Print header for function execution
    print("\n🎯 load_training_state")

    # Check for existence of saved model and state file
    state_path = model_ckpt_dir / "state.json"
    model_path = model_ckpt_dir / "latest.keras"

    if model_path.exists() and state_path.exists():
        with open(state_path, "r") as f:
            state = json.load(f)
        model = load_model(model_path)
        return model, state.get("initial_epoch", 0)

    return None, 0



# Function to train a model with checkpointing and optional resumption
def train_model(train_data, train_labels, model, model_name="mobilenet", verbose=1, result_file_path=None):
    """
    Trains a model using given data and logs best checkpoint metrics.

    Args:
        train_data (np.ndarray): Training images.
        train_labels (np.ndarray): One-hot encoded labels.
        model (tf.keras.Model): Compiled Keras model.
        model_name (str): Name of the model (used for checkpoint paths).
        verbose (int): Keras verbosity level.
        result_file_path (Path): Path to result.json file for storing logs.

    Returns:
        tuple: (trained model, Keras History object)
    """

    # Print header for function execution
    print("\n🎯 train_model")

    # Prepare output directory for model checkpoints
    model_ckpt_dir = CONFIG.CHECKPOINT_PATH / model_name
    model_ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Attempt to resume from checkpoint if available
    resumed_model, initial_epoch = load_training_state(model_ckpt_dir)
    if resumed_model:
        print(f"\n🔁 Resuming training from epoch {initial_epoch}")
        model = resumed_model
    else:
        initial_epoch = 0

    # Limit training set size (optional early-stage constraint)
    train_data = train_data[:5000]
    train_labels = train_labels[:5000]

    # Prepare all callbacks: model checkpoints + recovery
    callbacks = get_checkpoint_callbacks(model_ckpt_dir, verbose)
    callbacks.append(RecoveryCheckpoint(model_ckpt_dir))

    # Fit model on training data
    history = model.fit(
        x=train_data,
        y=train_labels,
        epochs=CONFIG.EPOCHS_COUNT,
        batch_size=CONFIG.BATCH_SIZE,
        validation_split=CONFIG.VALIDATION_SPLIT,
        callbacks=callbacks,
        verbose=verbose,
        initial_epoch=initial_epoch
    )

    # Log best result to JSON (if file path provided)
    best_val_acc = max(history.history.get("val_accuracy", [0]))
    if result_file_path:
        log_to_json(file_path=result_file_path, key="checkpoints", record={
            "model": model_name,
            "path": str(model_ckpt_dir / "best.keras"),
            "val_accuracy": round(float(best_val_acc), 4),
            "epochs": len(history.history.get("loss", [])),
            "status": "best"
        })

    return model, history



# Print confirmation message
print("\n✅ train.py successfully executed")
