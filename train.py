# Import standard libraries
import datetime
import json
from pathlib import Path
import pytz

# Import third-party libraries
from keras.api.callbacks import Callback, ModelCheckpoint
from keras.api.models import load_model


# Function to train a model
def train_model(train_data, train_labels, model, model_number, run, config_name, timestamp, config, verbose=2):
    """
    Trains a model using given data and logs all key metrics after training.

    If a checkpoint exists, training resumes from the last saved epoch.
    Stores final model and history to disk.

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
    print("\n🎯  train_model")

    # Create directory for model checkpointing
    model_checkpoint_path = config.CHECKPOINT_PATH / f"m{model_number}_r{run}_{config_name}"
    model_checkpoint_path.mkdir(parents=True, exist_ok=True)

    # Attempt to resume training from checkpoint if available
    resumed_model, initial_epoch, history = _resume_from_checkpoint(
        model_checkpoint_path, config, model_number, run, config_name
    )

    # If model was resumed and training is already complete, skip training
    if resumed_model is not None and history is None:
        return resumed_model, None, True  # Return early with resumed model and no new training

    # Use the resumed model if available
    if resumed_model is not None:
        model = resumed_model

    # Partition dataset into train/validation subsets
    train_data, train_labels, val_data, val_labels = _split_dataset(
        train_data, train_labels, config.LIGHT_MODE
    )

    # Prepare training callbacks (standard + custom recovery)
    callbacks = _prepare_checkpoint_callback(model_checkpoint_path, config)
    callbacks.append(RecoveryCheckpoint(model_checkpoint_path))

    try:
        # Only fit if no prior history recovered
        if history is None:
            model.model_id = model_number

            # Print training configuration to log file before training begins
            print("🧠  Printing training configuration:")
            print(f"Light Mode:       {'ON' if config.LIGHT_MODE else 'OFF'}")                                   # Light mode toggle
            print(f"Augmentation:     {'ON' if config.AUGMENT_MODE else 'OFF'}")                                 # Augmentation toggle

            print(f"L2 Regularization: {'ON' if config.L2_MODE['enabled'] else 'OFF'} (λ={config.L2_MODE['lambda']})")   # L2 setting
            print(f"Dropout:           {'ON' if config.DROPOUT_MODE['enabled'] else 'OFF'} (rate={config.DROPOUT_MODE['rate']})")  # Dropout setting

            print(f"Optimizer:         {config.OPTIMIZER['type']} (lr={config.OPTIMIZER['learning_rate']})")      # Optimizer and learning rate
            print(f"Momentum:          {config.OPTIMIZER.get('momentum', 0.0)}")                                  # Momentum value (if any)

            print(f"LR Scheduler:      {'ON' if config.SCHEDULE_MODE['enabled'] else 'OFF'}")                     # Learning rate scheduler toggle
            print(f"Early Stopping:    {'ON' if config.EARLY_STOP_MODE['enabled'] else 'OFF'}")                   # Early stopping toggle

            print(f"Epochs:            {config.EPOCHS_COUNT}")                                                    # Total training epochs
            print(f"Batch Size:        {config.BATCH_SIZE}\n")                                                    # Training batch size

            # Begin model training
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

            # Merge old history if training resumed
            if initial_epoch > 0:
                old_history_file = model_checkpoint_path / "history.json"
                if old_history_file.exists():
                    with open(old_history_file, "r") as f:
                        old_history = json.load(f)
                    for key in history.history:
                        history.history[key] = old_history.get(key, []) + history.history[key]


            # Save full history after training completes
            _save_training_history(model_checkpoint_path / "history.json", history)

    except Exception as e:
        # On failure, attempt to save partial history if available
        if hasattr(model, "history") and model.history:
            _save_training_history(model_checkpoint_path / "history.json", model.history)
        raise e

    # Save the trained model to disk
    model_path = config.MODEL_PATH / f"m{model_number}_r{run}_{config_name}.keras"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(model_path)

    return model, history, False  # Return fully trained model, history, and resumption flag


# Function to resume training from checkpoint if available
def _resume_from_checkpoint(checkpoint_path: Path, config, model_number: int, run: int, config_name: str):
    """
    Attempts to resume training from a saved checkpoint and training history.

    If a valid checkpoint and training history file are found, this function loads
    the saved model, restores the training epoch, and reattaches the history.

    Args:
        checkpoint_path (Path): Directory where checkpoint files are stored.
        config (Config): Configuration object with training parameters.
        model_number (int): Model identifier.
        run (int): Run ID number.
        config_name (str): Configuration name used for this run.

    Returns:
        tuple:
            - resumed_model (Model or None): Loaded Keras model if resume was possible.
            - initial_epoch (int): Epoch to resume from.
            - history (object or None): Dummy object with training history.
    """

    # Print header for function execution
    print("\n🎯  _resume_from_checkpoint")

    # Define path to the stored training history
    history_file = checkpoint_path / "history.json"

    # Load model and resume epoch if checkpoint exists
    resumed_model, initial_epoch = _load_from_checkpoint(checkpoint_path)
    history = None

    # Log resume status and handle early exit
    if resumed_model:
        print(f"\n🔁  Resuming experiment at epoch_{initial_epoch}")

        # If training was already completed, return early
        if initial_epoch >= config.EPOCHS_COUNT:
            print(f"\n⏩  Returning early from experiment m{model_number}_r{run}_{config_name}")
            return resumed_model, initial_epoch, None  # Early return: training complete

        # Attempt to load saved training history
        if history_file.exists():
            with open(history_file, "r") as f:
                history_data = json.load(f)
                class DummyHistory: pass
                history = DummyHistory()
                history.history = history_data  # Attach history data to dummy object

    return None if not resumed_model else resumed_model, initial_epoch, history  # Return resume state


# Function to split dataset
def _split_dataset(train_data, train_labels, light_mode):
    """
    Splits the dataset into training and validation sets.

    If light_mode is enabled, uses 20% of the dataset for validation.
    Otherwise, reserves the last 5000 samples.

    Args:
        train_data (np.ndarray): Input training data.
        train_labels (np.ndarray): Labels for the training data.
        light_mode (bool): If True, use 20% of data as validation;
                           otherwise use last 5000 samples.

    Returns:
        tuple: (train_data, train_labels, val_data, val_labels)
    """

    # Print header for function execution
    print("\n🎯  _split_dataset")

    # Determine split size and perform slicing
    if light_mode:
        val_split = int(0.2 * len(train_data))  # Use 20% for validation
        val_data = train_data[-val_split:]
        val_labels = train_labels[-val_split:]
        train_data = train_data[:-val_split]
        train_labels = train_labels[:-val_split]
    else:
        val_data = train_data[-5000:]  # Fixed-size validation set
        val_labels = train_labels[-5000:]
        train_data = train_data[:-5000]
        train_labels = train_labels[:-5000]

    return train_data, train_labels, val_data, val_labels  # Return split subsets


# Function to save training history
def _save_training_history(history_file: Path, history_obj):
    """
    Saves the Keras training history to a specified JSON file.

    Converts the History object to a dictionary and writes it to disk.

    Args:
        history_file (Path): Path to the history JSON file.
        history_obj (History): Keras History object containing training metrics.
    """

    # Print header for function execution
    print("\n🎯  _save_training_history")

    # Attempt to write training history to file
    try:
        with open(history_file, "w") as f:
            json.dump(history_obj.history, f)  # Serialize and write history data
    except Exception as e:
        print(f"\n⚠️ Failing to save history:\n{e}")  # Log failure if saving fails


# Class for saving model state
class RecoveryCheckpoint(Callback):
    """
    Custom Keras Callback that saves the model and a JSON state file after each epoch.

    This ensures that training progress can be resumed from the last saved state.

    Args:
        checkpoint_path (Path): Directory to store model and state.
    """

    def __init__(self, checkpoint_path: Path):
        """
        Constructor for the RecoveryCheckpoint callback.

        Sets up the checkpoint directory and target paths for saving the model
        and training state after each epoch.

        Args:
            checkpoint_path (Path): Directory to save model and state.
        """

        # Print header for constructor execution
        print("\n🎯  __init__ (RecoveryCheckpoint)\n")

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
        print("\n🎯  on_epoch_end")

        # Save model after this epoch
        self.model.save(self.model_path)

        # Write the current epoch to state.json
        with open(self.state_path, "w") as f:
            json.dump({"initial_epoch": epoch + 1}, f)

        # Confirm checkpoint write
        print(f"\n💾  Checkpointing experiment at epoch_{epoch + 1}\n")

        # Print timestamp for freeze detection
        print(f"🕒  Recording time at {datetime.datetime.now(pytz.timezone('Asia/Tehran')).strftime('%H:%M')}\n")


# Function to prepare checkpoint, LR scheduling, and early stopping callbacks
def _prepare_checkpoint_callback(model_checkpoint_path: Path, config):
    """
    Creates a list of Keras callbacks:
    - Saves the best model based on validation accuracy
    - Saves a model after each epoch
    - Reduces LR when validation accuracy plateaus (if SCHEDULE_MODE is enabled)
    - Stops training early if validation stagnates (if EARLY_STOP_MODE is enabled)

    Args:
        model_checkpoint_path (Path): Output directory for model checkpoints
        config (Config): Configuration object containing SCHEDULE_MODE and EARLY_STOP_MODE blocks

    Returns:
        list: A list of Keras callback objects
    """

    # Print header for function execution
    print("\n🎯  _prepare_checkpoint_callback")

    # Define checkpoint file paths
    best_model_path = model_checkpoint_path / "best.keras"               # best-performing model
    per_epoch_path = model_checkpoint_path / "epoch_{epoch:02d}.keras"   # all models per epoch

    # Extract verbosity levels for both scheduler and early stop
    verbose_lr = config.SCHEDULE_MODE.get("verbose", 1)
    verbose_es = config.EARLY_STOP_MODE.get("verbose", 1)

    # Initialize callback list with core checkpointing callbacks
    callbacks = [
        # Save only the best model based on validation accuracy
        ModelCheckpoint(
            filepath=best_model_path,
            monitor="val_accuracy",
            save_best_only=True,
            save_weights_only=False,
            verbose=verbose_lr
        ),

        # Save model at the end of every epoch regardless of performance
        ModelCheckpoint(
            filepath=per_epoch_path,
            save_best_only=False,
            save_weights_only=False,
            verbose=verbose_lr
        )
    ]

    # Append ReduceLROnPlateau if SCHEDULE_MODE is enabled
    if config.SCHEDULE_MODE.get("enabled", False):
        from keras.api.callbacks import ReduceLROnPlateau

        # Dynamically reduce learning rate when performance stagnates
        reduce_lr = ReduceLROnPlateau(
            monitor=config.SCHEDULE_MODE.get("monitor", "val_accuracy"),  # metric to watch
            factor=config.SCHEDULE_MODE.get("factor", 0.5),               # LR decay factor
            patience=config.SCHEDULE_MODE.get("patience", 3),             # how many stagnant epochs before decay
            min_lr=config.SCHEDULE_MODE.get("min_lr", 1e-5),              # LR floor
            verbose=verbose_lr                                            # verbosity
        )

        # Add LR scheduler to callback list
        callbacks.append(reduce_lr)

    # Append EarlyStopping if EARLY_STOP_MODE is enabled
    if config.EARLY_STOP_MODE.get("enabled", False):
        from keras.api.callbacks import EarlyStopping

        # Stop training early if no improvement after X epochs
        early_stop = EarlyStopping(
            monitor=config.EARLY_STOP_MODE.get("monitor", "val_accuracy"),          # metric to watch
            patience=config.EARLY_STOP_MODE.get("patience", 5),                     # how long to wait before stopping
            restore_best_weights=config.EARLY_STOP_MODE.get("restore_best_weights", True),  # reload best model
            verbose=verbose_es
        )

        # Add early stopping to callback list
        callbacks.append(early_stop)

    # Return the full list of callbacks
    return callbacks


# Function to resume model from checkpoint
def _load_from_checkpoint(model_checkpoint_path: Path):
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
    print("\n🎯  _load_from_checkpoint")

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
print("\n✅  train.py successfully executed")
