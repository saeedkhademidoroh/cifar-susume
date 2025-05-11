# Import standard libraries
import json
from pathlib import Path

# Import third-party libraries
from keras.api.callbacks import Callback, ModelCheckpoint
from keras.api.models import load_model

# Import project-specific libraries
from config import CONFIG
from log import log_to_json


# Function to train a model with checkpointing and optional resumption
def train_model(train_data, train_labels, model, model_number, verbose=2):
    """
    Trains a model using given data and logs all key metrics after training.

    Args:
        train_data (np.ndarray): Training images.
        train_labels (np.ndarray): Integer class labels.
        model (tf.keras.Model): Compiled Keras model.
        model_number (int): Model version number.
        verbose (int): Keras verbosity level.

    Returns:
        tuple: (trained model, Keras History object)
    """

    print("\n🎯 train_model")

    model_checkpoint_path = CONFIG.CHECKPOINT_PATH / f"m{model_number}"
    model_checkpoint_path.mkdir(parents=True, exist_ok=True)

    if not CONFIG.CLEAN_CHECKPOINT:
        resumed_model, initial_epoch = load_training_state(model_checkpoint_path)
        if resumed_model:
            print(f"\n🔁 Resumed: epoch_{initial_epoch}")
            model = resumed_model
        else:
            initial_epoch = 0
    else:
        initial_epoch = 0


    if CONFIG.LIGHT_MODE:
        # In light mode, use 20% of reduced train set for validation
        val_split = int(0.2 * len(train_data))
        val_data = train_data[-val_split:]
        val_labels = train_labels[-val_split:]
        train_data = train_data[:-val_split]
        train_labels = train_labels[:-val_split]
    else:
        # In full mode, use fixed 5,000-sample validation set
        val_data = train_data[-5000:]
        val_labels = train_labels[-5000:]
        train_data = train_data[:-5000]
        train_labels = train_labels[:-5000]

    callbacks = get_checkpoint_callbacks(model_checkpoint_path, verbose)
    callbacks.append(RecoveryCheckpoint(model_checkpoint_path))

    history = model.fit(
        x=train_data,
        y=train_labels,
        validation_data=(val_data, val_labels),
        epochs=CONFIG.EPOCHS_COUNT,
        batch_size=CONFIG.BATCH_SIZE,
        callbacks=callbacks,
        verbose=verbose,
        initial_epoch=initial_epoch
    )

    return model, history


# Function to define a custom recovery callback that saves the model and state after each epoch
class RecoveryCheckpoint(Callback):
    """
    Custom Keras callback to save a recovery checkpoint after each epoch.
    Saves both the model and a state.json file with current epoch.
    """

    # Constructor to prepare checkpoint directory and file paths
    def __init__(self, checkpoint_path: Path):
        """
        Initializes recovery paths and ensures checkpoint folder exists.
        """

        # Print header for function execution
        print("\n🎯 __init__ (RecoveryCheckpoint)")

        super().__init__()
        self.checkpoint_path = checkpoint_path
        self.checkpoint_path.mkdir(parents=True, exist_ok=True)
        self.model_path = checkpoint_path / "latest.keras"
        self.state_path = checkpoint_path / "state.json"


    # Function to save model and state.json after each epoch
    def on_epoch_end(self, epoch, logs=None):
        """
        Saves model and training state at the end of each epoch.
        """

        # Print header for function execution
        print("\n🎯 on_epoch_end")

        self.model.save(self.model_path)
        with open(self.state_path, "w") as f:
            json.dump({"initial_epoch": epoch + 1}, f)
        print(f"\n💾 Checkpoint: epoch_{epoch + 1}\n")


# Function to prepare ModelCheckpoint callbacks
def get_checkpoint_callbacks(model_checkpoint_path: Path, verbose: int):
    """
    Returns a list of Keras callbacks to save both best and per-epoch checkpoints.
    """

    # Print header for function execution
    print("\n🎯 get_checkpoint_callbacks")

    # Define file paths for best and epoch-based checkpoints
    best_model_path = model_checkpoint_path / "best.keras"
    per_epoch_path = model_checkpoint_path / "epoch_{epoch:02d}.keras"

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
            verbose=verbose
        )
    ]


# Function to resume training state from checkpoint if available
def load_training_state(model_checkpoint_path: Path):
    """
    Loads model and epoch state from previous checkpoint if available.

    Returns:
        tuple: (loaded model or None, starting epoch)
    """

    # Print header for function execution
    print("\n🎯 load_training_state")

    # Check for existence of saved model and state file
    state_path = model_checkpoint_path / "state.json"
    model_path = model_checkpoint_path / "latest.keras"

    if model_path.exists() and state_path.exists():
        with open(state_path, "r") as f:
            state = json.load(f)
        model = load_model(model_path)
        return model, state.get("initial_epoch", 0)

    return None, 0


# Print confirmation message
print("\n✅ train.py successfully executed")
