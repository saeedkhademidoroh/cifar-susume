# Import standard libraries
import datetime
import json
from pathlib import Path
from types import SimpleNamespace

# Import third-party libraries
import pytz
import tensorflow as tf
from tensorflow.python.client import device_lib
from keras.api.losses import CategoricalCrossentropy
from keras.api.metrics import Mean, SparseCategoricalAccuracy
from keras.api.callbacks import Callback, ModelCheckpoint, EarlyStopping
from keras.api.models import load_model


# Function to print training context
def _print_training_context(config):
    """
    Prints available compute devices and training configuration details.

    Args:
        config (Config): Loaded configuration object.
    """

    # Print header for function execution
    print("\n🎯  _print_training_context")

    # Print available devices
    print("\n🖥️   Available compute devices:")
    for device in device_lib.list_local_devices():
        print(f"  • {device.name} ({device.device_type})")

    gpus = tf.config.list_physical_devices("GPU")
    print(f"\n🧮  GPU detected: {len(gpus) > 0}")
    for gpu in gpus:
        print(f"  • {gpu.name}")

    # Print configuration summary
    print("\n🧠  Printing training configuration:")
    print(f"Light Mode:         {'ON' if config.LIGHT_MODE else 'OFF'} — Using reduced dataset for fast testing")

    print(f"Augmentation:       {'ON' if config.AUGMENT_MODE['enabled'] else 'OFF'}", end="")
    if config.AUGMENT_MODE['enabled']:
        print(" —", end=" ")
        flags = []
        if config.AUGMENT_MODE.get("random_crop", False):
            flags.append("Random Crop")
        if config.AUGMENT_MODE.get("random_flip", False):
            flags.append("Horizontal Flip")
        if config.AUGMENT_MODE.get("cutout", False):
            flags.append("Cutout")
        if config.AUGMENT_MODE.get("color_jitter", False):
            flags.append("Color Jitter")
        print(", ".join(flags))
    else:
        print()

    print(f"L2 Regularization:  {'ON' if config.L2_MODE['enabled'] else 'OFF'} (λ = {config.L2_MODE['lambda']})")
    print(f"Dropout:            {'ON' if config.DROPOUT_MODE['enabled'] else 'OFF'} (rate = {config.DROPOUT_MODE['rate']})")
    print(f"Optimizer:          {config.OPTIMIZER['type'].upper()} (lr = {config.OPTIMIZER['learning_rate']})")
    print(f"Momentum:           {config.OPTIMIZER.get('momentum', 0.0)}")

    print(f"LR Scheduler:       {'ON' if config.SCHEDULE_MODE['enabled'] else 'OFF'}", end="")
    if config.SCHEDULE_MODE['enabled']:
        print(f" — warmup for {config.SCHEDULE_MODE.get('warmup_epochs', 0)} epochs, decay factor {config.SCHEDULE_MODE.get('gamma', '?')}")
    else:
        print()

    print(f"Early Stopping:     {'ON' if config.EARLY_STOP_MODE['enabled'] else 'OFF'}", end="")
    if config.EARLY_STOP_MODE['enabled']:
        print(f" — patience {config.EARLY_STOP_MODE.get('patience', '?')} epochs, restore best weights: {config.EARLY_STOP_MODE.get('restore_best_weights', False)}")
    else:
        print()

    print(f"Weight Averaging:   {'ON' if config.AVERAGE_MODE['enabled'] else 'OFF'}", end="")
    if config.AVERAGE_MODE['enabled']:
        print(f" — starting at epoch {config.AVERAGE_MODE.get('start_epoch', '?')}")
    else:
        print()

    print(f"Test-Time Augment:  {'ON' if config.TTA_MODE['enabled'] else 'OFF'}", end="")
    if config.TTA_MODE['enabled']:
        print(f" — running {config.TTA_MODE.get('runs', 1)} augmented passes per sample")
    else:
        print()

    print(f"MixUp:              {'ON' if config.MIXUP_MODE['enabled'] else 'OFF'}", end="")
    if config.MIXUP_MODE['enabled']:
        print(f" — alpha = {config.MIXUP_MODE.get('alpha', '?')}")
    else:
        print()

    print(f"Epochs:             {config.EPOCHS_COUNT}")
    print(f"Batch Size:         {config.BATCH_SIZE}\n")


# Function to train a model
def train_model(train_data, train_labels, model, model_number, run, config_name, config, verbose=2):
    """
    Trains a model using a custom training loop and logs key training metrics.

    Resumes training from the last saved checkpoint if available.
    Supports advanced training strategies (e.g., MixUp, SWA).
    Saves the final model and full training history to disk.
    """


    # Print header for function execution
    print("\n🎯  train_model")

    # Define and create model checkpoint directory for this run
    run_id = f"m{model_number}_r{run}_{config_name}"
    model_checkpoint_path = config.CHECKPOINT_PATH / run_id
    model_checkpoint_path.mkdir(parents=True, exist_ok=True)

    # Attempt to resume model, epoch, and history from previous checkpoint
    resumed_model, initial_epoch, history = _resume_from_checkpoint(run_id, config)

    # Skip training if model is already complete
    if resumed_model and initial_epoch >= config.EPOCHS_COUNT:
        print(f"\n📦  Training already complete — skipping m{model_number}_r{run}_{config_name}")
        return resumed_model, None, True


    # If training is incomplete but a fake or partial history exists, discard it
    if resumed_model is not None and initial_epoch < config.EPOCHS_COUNT and history is not None:
        print("\n⚠️  Continuing training and rebuilding partial history")
        history = None  # Force retraining from the resumed model

    # Use resumed model if available
    if resumed_model is not None:
        model = resumed_model

    # Split dataset into training and validation sets
    train_data, train_labels, val_data, val_labels = _split_dataset(
        train_data, train_labels, config.LIGHT_MODE
    )

    # Build tf.data pipeline for training batches
    train_dataset = tf.data.Dataset.from_tensor_slices((train_data, train_labels))
    train_dataset = train_dataset.shuffle(1024).batch(config.BATCH_SIZE)

    # Prepare checkpoint, scheduler, and early stop callbacks
    callbacks = _prepare_callback(model, model_checkpoint_path, config)
    callbacks.append(RecoveryCheckpoint(model_checkpoint_path))

    # Always assign model metadata
    model.model_id = model_number
    model.run_id = f"m{model_number}_r{run}_{config_name}"

    try:
        # Only fit if no prior history was recovered
        if history is None:

            # Print system info and experiment hyperparameters for traceability
            _print_training_context(config)

            # Extract MixUp settings from config
            mixup_enabled = config.MIXUP_MODE.get("enabled", False)
            mixup_alpha = config.MIXUP_MODE.get("alpha", 0.2)

            # Cache training data as tensors (needed for MixUp)
            train_data_tensor = tf.convert_to_tensor(train_data)
            train_labels_tensor = tf.convert_to_tensor(train_labels)

            # Build training dataset
            train_dataset = tf.data.Dataset.from_tensor_slices((train_data, train_labels))
            train_dataset = train_dataset.shuffle(1024)

            if mixup_enabled:
                def dataset_mixup(x, y):
                    return _mixup_fn(x, y, train_data_tensor, train_labels_tensor, alpha=mixup_alpha)
                train_dataset = train_dataset.map(dataset_mixup, num_parallel_calls=tf.data.AUTOTUNE)
            else:
                def one_hot_encode(x, y):
                    return x, tf.one_hot(y, 10)
                train_dataset = train_dataset.map(one_hot_encode, num_parallel_calls=tf.data.AUTOTUNE)

            train_dataset = train_dataset.batch(config.BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

            # Initialize custom history tracking
            history = {"loss": [], "accuracy": [], "val_loss": [], "val_accuracy": []}

            for epoch in range(initial_epoch, config.EPOCHS_COUNT):
                print(f"\n🔁 Epoch {epoch + 1}/{config.EPOCHS_COUNT}\n")
                epoch_loss = Mean()
                epoch_acc = SparseCategoricalAccuracy()

                for x_batch, y_batch in train_dataset:
                    with tf.GradientTape() as tape:
                        preds = model(x_batch, training=True)
                        loss = CategoricalCrossentropy()(y_batch, preds)

                    grads = tape.gradient(loss, model.trainable_variables)
                    model.optimizer.apply_gradients(zip(grads, model.trainable_variables))

                    epoch_loss.update_state(loss)
                    epoch_acc.update_state(tf.argmax(y_batch, axis=1), preds)

                print(f"📊 Epoch {epoch + 1} — Loss: {epoch_loss.result():.4f} — Acc: {epoch_acc.result():.4f}")

                history["loss"].append(epoch_loss.result().numpy())
                history["accuracy"].append(epoch_acc.result().numpy())

            # Merge old history if training resumed
            if initial_epoch > 0:
                old_history_file = model_checkpoint_path / "history.json"
                if old_history_file.exists():
                    with open(old_history_file, "r") as f:
                        old_history = json.load(f)
                    for key in history:
                        history[key] = old_history.get(key, []) + history[key]

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


# Function to prepare callback
def _prepare_callback(model, model_checkpoint_path: Path, config):
    """
    Creates a list of Keras callbacks:
    - Saves the best model based on validation accuracy
    - Saves a model after each epoch
    - Applies milestone-based StepLR decay (if SCHEDULE_MODE is enabled)
    - Applies optional learning rate warmup at start of training
    - Stops training early if validation stagnates (if EARLY_STOP_MODE is enabled)
    """

    # Print header for function execution
    print("\n🎯  _prepare_checkpoint_callback")

    # Define checkpoint file paths
    best_model_path = model_checkpoint_path / "best.keras"
    per_epoch_path = model_checkpoint_path / "epoch_{epoch:02d}.keras"

    # Extract verbosity settings from config
    verbose_lr = config.SCHEDULE_MODE.get("verbose", 1)
    verbose_es = config.EARLY_STOP_MODE.get("verbose", 1)

    # Initialize core checkpoint callbacks:
    # - Best model (based on val_accuracy)
    # - Model after every epoch
    callbacks = [
        ModelCheckpoint(
            filepath=best_model_path,
            monitor="val_accuracy",
            save_best_only=True,
            save_weights_only=False,
            verbose=verbose_lr
        ),
        ModelCheckpoint(
            filepath=per_epoch_path,
            save_best_only=False,
            save_weights_only=False,
            verbose=verbose_lr
        )
    ]

    # Conditionally append EarlyStopping if enabled in config
    if config.EARLY_STOP_MODE.get("enabled", False):
        early_stop = EarlyStopping(
            monitor=config.EARLY_STOP_MODE.get("monitor", "val_accuracy"),
            patience=config.EARLY_STOP_MODE.get("patience", 5),
            restore_best_weights=config.EARLY_STOP_MODE.get("restore_best_weights", True),
            verbose=verbose_es
        )
        callbacks.append(early_stop)

    # Scheduler logic based on config.SCHEDULE_MODE
    if config.SCHEDULE_MODE.get("enabled", False):
        scheduler_type = config.SCHEDULE_MODE.get("type", "").lower()

        # StepLR scheduler (milestone-based learning rate decay)
        if scheduler_type == "step":
            # Custom Keras callback to implement milestone-based LR decay
            class StepDecayScheduler(Callback):
                def __init__(self, optimizer, initial_lr, milestones, gamma):
                    super().__init__()
                    self.optimizer = optimizer
                    self.initial_lr = initial_lr
                    self.milestones = milestones
                    self.gamma = gamma

                def on_epoch_begin(self, epoch, logs=None):
                    # Compute decayed LR based on number of milestones passed
                    lr = self.initial_lr
                    for milestone in self.milestones:
                        if epoch >= milestone:
                            lr *= self.gamma
                    # Apply new learning rate
                    self.model.optimizer.learning_rate.assign(lr)
                    print(f"\n🔁  StepLR applied — epoch {epoch}, learning rate set to {lr:.5f}\n")

            # Instantiate StepDecayScheduler with config values
            step_scheduler = StepDecayScheduler(
                optimizer=model.optimizer,
                initial_lr=config.OPTIMIZER.get("learning_rate", 0.1),
                milestones=config.SCHEDULE_MODE.get("milestones", [80, 120]),
                gamma=config.SCHEDULE_MODE.get("gamma", 0.1)
            )
            callbacks.append(step_scheduler)

        # ReduceLROnPlateau scheduler (metric-monitored adaptive decay)
        elif scheduler_type == "plateau":
            from keras.api.callbacks import ReduceLROnPlateau
            scheduler = ReduceLROnPlateau(
                monitor=config.SCHEDULE_MODE.get("monitor", "val_accuracy"),
                factor=config.SCHEDULE_MODE.get("factor", 0.5),
                patience=config.SCHEDULE_MODE.get("patience", 5),
                min_lr=config.SCHEDULE_MODE.get("min_lr", 1e-5),
                verbose=config.SCHEDULE_MODE.get("verbose", 1)
            )
            callbacks.append(scheduler)

        # Linear warmup scheduler (optional, applies during early epochs)
        if config.SCHEDULE_MODE.get("warmup_epochs", 0) > 0:
            # Custom Keras callback to linearly ramp up LR for N warmup epochs
            class LinearWarmupScheduler(Callback):
                def __init__(self, optimizer, warmup_epochs, target_lr):
                    super().__init__()
                    self.optimizer = optimizer
                    self.warmup_epochs = warmup_epochs
                    self.target_lr = target_lr

                def on_epoch_begin(self, epoch, logs=None):
                    if epoch < self.warmup_epochs:
                        # Linearly scale LR based on warmup progress
                        warmup_lr = self.target_lr * ((epoch + 1) / self.warmup_epochs)
                        self.model.optimizer.learning_rate.assign(warmup_lr)
                        print(f"\n🔥  WarmupLR applied — epoch {epoch}, learning rate set to {warmup_lr:.5f}\n")

            # Instantiate LinearWarmupScheduler with target LR and warmup period
            warmup_scheduler = LinearWarmupScheduler(
                optimizer=model.optimizer,
                warmup_epochs=config.SCHEDULE_MODE.get("warmup_epochs"),
                target_lr=config.OPTIMIZER.get("learning_rate", 0.1)
            )
            callbacks.append(warmup_scheduler)

    # Return all composed callbacks for model.fit()
    # This includes checkpointing, early stopping, LR scheduling, and warmup (if enabled)
    return callbacks


# Class for saving model state
class RecoveryCheckpoint(Callback):
    """
    Custom Keras callback to save model and epoch state after every epoch.

    This enables training to be resumed precisely from where it left off.
    It writes the model to `latest.keras` and training progress to `state.json`.

    Args:
        checkpoint_path (Path): Directory where model and state will be stored.
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


# Function to resume training from checkpoint if available
def _resume_from_checkpoint(run_id: str, config):
    """
    Attempts to resume training from a saved checkpoint and training history.

    Checks if a model checkpoint and history file exist for the given run_id.
    If available, restores the model and resumes from the last saved epoch,
    reconstructing a dummy History-like object for compatibility.

    Args:
        run_id (str): Unique identifier for the experiment (e.g. "m9_r1_default").
        config (Config): Configuration object with checkpoint and history paths.

    Returns:
        tuple:
            - resumed_model (keras.Model or None): Restored model, or None if not found.
            - initial_epoch (int): Epoch to resume training from (defaults to 0).
            - history (object or None): Dummy object simulating Keras History, or None.
    """

    # Print header for function execution
    print("\n🎯  _resume_from_checkpoint")

    # Define path to the stored training history
    checkpoint_path = config.CHECKPOINT_PATH / run_id
    history_file = checkpoint_path / "history.json"

    # Load model and resume epoch if checkpoint exists
    resumed_model, initial_epoch = _load_from_checkpoint(checkpoint_path)
    history = None

    # Log resume status and handle early exit
    if resumed_model:
        print(f"\n🔁  Resuming experiment {run_id} at epoch_{initial_epoch}")

        # If training was already completed, return early
        if initial_epoch >= config.EPOCHS_COUNT:
            print(f"\n⏩  Returning early from experiment {run_id}")
            return resumed_model, initial_epoch, None  # Early return: training complete

        # Attempt to load saved training history
        if history_file.exists():
            with open(history_file, "r") as f:
                history_data = json.load(f)
            history = SimpleNamespace(history=history_data)  # Wrap dict in object with .history attribute

    return None if not resumed_model else resumed_model, initial_epoch, history  # Return resume state


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


# Function to apply MixUp augmentation to a training batch inside a tf.data pipeline
def _mixup_fn(x1, y1, train_data_tensor, train_labels_tensor, alpha=0.2):
    """
    Applies MixUp augmentation on the given input batch by linearly combining each sample
    with a randomly selected counterpart from the dataset.

    Args:
        x1 (tf.Tensor): A batch of input images of shape [B, H, W, C].
        y1 (tf.Tensor): A batch of integer labels of shape [B].
        train_data_tensor (tf.Tensor): Full training dataset tensor [N, H, W, C] for sampling.
        train_labels_tensor (tf.Tensor): Corresponding labels tensor [N].
        alpha (float): MixUp alpha parameter controlling interpolation strength.

    Returns:
        tuple: (x_mix, y_mix)
            x_mix (tf.Tensor): Batch of mixed images.
            y_mix (tf.Tensor): Batch of soft labels (one-hot interpolated).
    """


    # Print header for function execution
    print("\n🎯  _mixup_fn")

    # Randomly select indices from the full dataset to pair with current batch
    idx = tf.random.shuffle(tf.range(tf.shape(train_data_tensor)[0]))[:tf.shape(x1)[0]]

    # Gather alternative samples
    x2 = tf.gather(train_data_tensor, idx)
    y2 = tf.gather(train_labels_tensor, idx)

    # Sample lambda from Beta(alpha, alpha) (approximated here as uniform)
    lam = tf.random.uniform([], 1 - alpha, 1.0)

    # Mix images and labels
    x_mix = lam * x1 + (1 - lam) * x2
    y_mix = lam * tf.one_hot(y1, 10) + (1 - lam) * tf.one_hot(y2, 10)

    return x_mix, y_mix


# Function to save training history
def _save_training_history(history_file: Path, history_obj):
    """
    Saves training history to a JSON file.

    Extracts the `.history` dictionary from a Keras History-like object
    and writes it to disk as a JSON file.

    Args:
        history_file (Path): Path to the history JSON file.
        history_obj: Object with a `.history` attribute (typically Keras History).
    """


    # Print header for function execution
    print("\n🎯  _save_training_history")

    # Attempt to write training history to file
    try:
        with open(history_file, "w") as f:
            json.dump(history_obj.history, f)  # Serialize and write history data
    except Exception as e:
        print(f"\n⚠️  Failing to save history:\n{e}")  # Log failure if saving fails


# Print confirmation message
print("\n✅  train.py successfully executed")
