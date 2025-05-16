# Import standard libraries
import json
from pathlib import Path

# Import third-party libraries
from keras.api.callbacks import Callback, ModelCheckpoint
from keras.api.models import load_model


# Function to train a model with checkpointing and optional resumption
def train_model(train_data, train_labels, model, model_number, run, config_name, timestamp, config, verbose=2):
    """
    Trains a model using given data and logs all key metrics after training.

    Args:
        train_data (np.ndarray): Training images.
        train_labels (np.ndarray): Integer class labels.
        model (tf.keras.Model): Compiled Keras model.
        model_number (int): Model version number.
        run (int): Run number.
        config_name (str): Name of the configuration used.
        timestamp (str): Unique timestamp for this training session.
        config (Config): Dynamic configuration object.
        verbose (int): Keras verbosity level.

    Returns:
        tuple: (trained model, Keras History object, was resumed)
    """

    # Print header for function execution
    print("\n🎯 train_model")

    # Create model checkpoint directory for saving progress and results
    model_checkpoint_path = config.CHECKPOINT_PATH / f"m{model_number}_r{run}_{config_name}"
    model_checkpoint_path.mkdir(parents=True, exist_ok=True)

    # Attempt to load model and training history if a checkpoint exists
    resumed_model, initial_epoch, history = _resume_from_checkpoint(
        model_checkpoint_path, config, model_number, run, config_name
    )

    # If training was already completed (resumed model with no need to train), return early
    if resumed_model is not None and history is None:
        return resumed_model, None, True

    # If a model was successfully resumed, use it; otherwise keep the passed-in model
    if resumed_model is not None:
        model = resumed_model

    # Partition dataset into training and validation sets
    train_data, train_labels, val_data, val_labels = _split_train_validation(
        train_data, train_labels, config.LIGHT_MODE
    )

    # Prepare list of training callbacks
    callbacks = get_checkpoint_callbacks(model_checkpoint_path, verbose)
    callbacks.append(RecoveryCheckpoint(model_checkpoint_path))

    # Begin training only if history was not previously resumed
    try:
        if history is None:
            model.model_id = model_number
            history = model.fit(
                x=train_data,
                y=train_labels,
                validation_data=(val_data, val_labels),
                epochs=config.EPOCHS_COUNT,
                batch_size=config.BATCH_SIZE,
                callbacks=callbacks,
                verbose=verbose,
                initial_epoch=initial_epoch
            )
            # Save training history after successful training
            _save_training_history(model_checkpoint_path / "history.json", history)

    # If training fails mid-run, save partial history if available
    except Exception as e:
        if hasattr(model, "history") and model.history:
            _save_training_history(model_checkpoint_path / "history.json", model.history)
        raise e

    # Save final trained model
    model_path = config.MODEL_PATH / f"m{model_number}_r{run}_{config_name}.keras"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(model_path)

    return model, history, False


# Resume training from checkpoint if available
def _resume_from_checkpoint(checkpoint_path: Path, config, model_number: int, run: int, config_name: str):
    history_file = checkpoint_path / "history.json"
    resumed_model, initial_epoch = load_training_state(checkpoint_path)
    history = None

    # Log resume status
    if resumed_model:
        print(f"\n🔁 Resumed: epoch_{initial_epoch}")
        if initial_epoch >= config.EPOCHS_COUNT:
            print(f"\n⏩ Training already completed for m{model_number}_r{run}_{config_name}")
            return resumed_model, initial_epoch, None

        # Try loading history
        if history_file.exists():
            with open(history_file, "r") as f:
                history_data = json.load(f)
                class DummyHistory: pass
                history = DummyHistory()
                history.history = history_data

    return None, initial_epoch, history


# Helper function: split dataset into training and validation sets
def _split_train_validation(train_data, train_labels, light_mode):
    """
    Splits the dataset into training and validation sets.

    Args:
        train_data (np.ndarray): Input training data.
        train_labels (np.ndarray): Labels for the training data.
        light_mode (bool): If True, use 20% of data as validation; otherwise use last 5000 samples.

    Returns:
        tuple: (train_data, train_labels, val_data, val_labels)
    """

    # Determine validation size based on LIGHT_MODE
    if light_mode:
        val_split = int(0.2 * len(train_data))
        val_data = train_data[-val_split:]
        val_labels = train_labels[-val_split:]
        train_data = train_data[:-val_split]
        train_labels = train_labels[:-val_split]
    else:
        val_data = train_data[-5000:]
        val_labels = train_labels[-5000:]
        train_data = train_data[:-5000]
        train_labels = train_labels[:-5000]

    # Return split dataset
    return train_data, train_labels, val_data, val_labels


# Helper function: save history object to disk
def _save_training_history(history_file: Path, history_obj):
    """
    Saves the Keras training history to a specified JSON file.

    Args:
        history_file (Path): Path to the history JSON file.
        history_obj (History): Keras History object containing training metrics.
    """

    # Attempt to write training history to file
    try:
        with open(history_file, "w") as f:
            json.dump(history_obj.history, f)
    except Exception as e:
        print(f"\n⚠️ Failed to save history:\n{e}")


# Callback class to save model and state.json after each epoch
class RecoveryCheckpoint(Callback):
    """
    Custom Keras Callback that saves the model and a JSON state file after each epoch.
    Used to enable training resumption from the last completed epoch.

    Args:
        checkpoint_path (Path): Directory to store model and state.
    """

    def __init__(self, checkpoint_path: Path):
        # Print header for constructor execution
        print("\n🎯 __init__ (RecoveryCheckpoint)\n")

        # Initialize the base Callback class
        super().__init__()

        # Create checkpoint directory if it doesn't exist
        self.checkpoint_path = checkpoint_path
        self.checkpoint_path.mkdir(parents=True, exist_ok=True)

        # File to save the model after each epoch
        self.model_path = checkpoint_path / "latest.keras"

        # File to track current training epoch
        self.state_path = checkpoint_path / "state.json"


    def on_epoch_end(self, epoch, logs=None):
        """
        Callback method executed at the end of every training epoch.
        Saves the current model and the epoch number.

        Args:
            epoch (int): Current epoch number (0-based).
            logs (dict): Metrics from this epoch.
        """

        # Print execution header for epoch end
        print("\n🎯 on_epoch_end")

        # Save model after this epoch
        self.model.save(self.model_path)

        # Write the current epoch to state.json
        with open(self.state_path, "w") as f:
            json.dump({"initial_epoch": epoch + 1}, f)

        # Confirm checkpoint write
        print(f"\n💾 Checkpoint: epoch_{epoch + 1}\n")


# Function to prepare callbacks that save best model and per-epoch models
def get_checkpoint_callbacks(model_checkpoint_path: Path, verbose: int):
    """
    Creates a list of Keras ModelCheckpoint callbacks:
      - One for saving the best model based on validation accuracy.
      - One for saving a model at the end of every epoch.

    Args:
        model_checkpoint_path (Path): Directory where models will be saved.
        verbose (int): Verbosity level for checkpoint logging.

    Returns:
        list: A list of Keras callback objects.
    """

    # Print header for function execution
    print("\n🎯 get_checkpoint_callbacks")

    # Define file paths for saving best and per-epoch models
    best_model_path = model_checkpoint_path / "best.keras"
    per_epoch_path = model_checkpoint_path / "epoch_{epoch:02d}.keras"

    # Return list of ModelCheckpoint callbacks
    return [
        # Save the best model according to validation accuracy
        ModelCheckpoint(
            filepath=best_model_path,
            monitor='val_accuracy',
            save_best_only=True,
            save_weights_only=False,
            verbose=verbose
        ),

        # Save a model at the end of every epoch
        ModelCheckpoint(
            filepath=per_epoch_path,
            save_best_only=False,
            save_weights_only=False,
            verbose=verbose
        )
    ]


# Function to resume model and epoch number from previously saved checkpoint
def load_training_state(model_checkpoint_path: Path):
    """
    Attempts to resume training by loading the latest saved model and training state.

    Args:
        model_checkpoint_path (Path): Directory containing checkpoint files.

    Returns:
        tuple: A tuple containing:
            - model (tf.keras.Model or None): Loaded model if available, otherwise None.
            - initial_epoch (int): Epoch to resume training from. Defaults to 0 if not available.
    """

    # Print header for function execution
    print("\n🎯 load_training_state")

    # Define paths to the saved model and training state
    state_path = model_checkpoint_path / "state.json"
    model_path = model_checkpoint_path / "latest.keras"

    # Load model and state if both files exist
    if model_path.exists() and state_path.exists():
        # Load training metadata (e.g., last completed epoch)
        with open(state_path, "r") as f:
            state = json.load(f)

        # Load the saved model
        model = load_model(model_path)

        # Return resumed model and the epoch to resume from
        return model, state.get("initial_epoch", 0)

    # Return default values if checkpoint files are not found
    return None, 0



# Print confirmation message
print("\n✅ train.py successfully executed")
