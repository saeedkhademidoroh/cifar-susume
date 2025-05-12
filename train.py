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

    # Define model-specific checkpoint directory
    model_checkpoint_path = CONFIG.CHECKPOINT_PATH / f"m{model_number}"
    model_checkpoint_path.mkdir(parents=True, exist_ok=True)

    # Optionally resume training from saved checkpoint
    if not CONFIG.CLEAN_CHECKPOINT:
        resumed_model, initial_epoch = load_training_state(model_checkpoint_path)
        if resumed_model:
            print(f"\n🔁 Resumed: epoch_{initial_epoch}")
            model = resumed_model
        else:
            initial_epoch = 0
    else:
        initial_epoch = 0

    # Validation split logic based on mode
    if CONFIG.LIGHT_MODE:
        # Use 20% of reduced train set as validation in light mode
        val_split = int(0.2 * len(train_data))
        val_data = train_data[-val_split:]
        val_labels = train_labels[-val_split:]
        train_data = train_data[:-val_split]
        train_labels = train_labels[:-val_split]
    else:
        # Use fixed 5,000 samples for validation in full mode
        val_data = train_data[-5000:]
        val_labels = train_labels[-5000:]
        train_data = train_data[:-5000]
        train_labels = train_labels[:-5000]

    # Setup callbacks for checkpointing
    callbacks = get_checkpoint_callbacks(model_checkpoint_path, verbose)
    callbacks.append(RecoveryCheckpoint(model_checkpoint_path))

    # Train the model with validation and checkpointing
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

    # Save final model to disk for evaluation and future use
    CONFIG.MODEL_PATH.mkdir(parents=True, exist_ok=True)
    model_path = CONFIG.MODEL_PATH / f"m{model_number}.keras"
    model.save(model_path)


    # Return the trained model and history
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

        # Initialize the parent Callback class
        super().__init__()

        # Store the checkpoint directory for this model
        self.checkpoint_path = checkpoint_path

        # Ensure the checkpoint directory exists
        self.checkpoint_path.mkdir(parents=True, exist_ok=True)

        # Path to the latest model file (used for recovery)
        self.model_path = checkpoint_path / "latest.keras"

        # Path to JSON file for storing training state (e.g., epoch)
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
        # Callback to save only the best-performing model based on validation accuracy
        ModelCheckpoint(
            filepath=best_model_path,
            monitor='val_accuracy',
            save_best_only=True,
            save_weights_only=False,
            verbose=verbose
        ),

        # Callback to save a model checkpoint at the end of every epoch
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

    # If both model file and training state file exist, resume training
    if model_path.exists() and state_path.exists():
        # Load training metadata (e.g., last completed epoch)
        with open(state_path, "r") as f:
            state = json.load(f)

        # Load the saved model
        model = load_model(model_path)

        # Return resumed model and initial_epoch for continuation
        return model, state.get("initial_epoch", 0)

    # If no valid checkpoint found, return None and start from epoch 0
    return None, 0



# Print confirmation message
print("\n✅ train.py successfully executed")
