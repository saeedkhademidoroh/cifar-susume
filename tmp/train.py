# Import standard libraries
import datetime
import json
from pathlib import Path
from types import SimpleNamespace

# Import third-party libraries
import pytz
import tensorflow as tf
from keras.api.losses import SparseCategoricalCrossentropy
from keras.api.metrics import Mean, SparseCategoricalAccuracy
from keras.api.callbacks import Callback, ModelCheckpoint, EarlyStopping

# Import project-specific libraries
from utility import load_from_checkpoint, save_training_history


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

    # Step 0: Print header for function execution
    print("\n🎯  _resume_from_checkpoint")

    # Step 1: Compute checkpoint path and locate history file
    checkpoint_path = config.CHECKPOINT_PATH / run_id
    history_file = checkpoint_path / "history.json"

    # Step 2: Try to load model and state from disk
    resumed_model, initial_epoch = load_from_checkpoint(checkpoint_path)
    history = None

    # Step 3: If resume is successful, print log and check for completion
    if resumed_model:
        print(f"\n🔁  Resuming experiment {run_id} at epoch_{initial_epoch}")

        # Step 4: If already completed, skip training
        if initial_epoch >= config.EPOCHS_COUNT:
            print(f"\n⏩  Returning early from experiment {run_id}")
            return resumed_model, initial_epoch, None

        # Step 5: If not done, try to recover old training history
        if history_file.exists():
            with open(history_file, "r") as f:
                history_data = json.load(f)
            history = SimpleNamespace(history=history_data)

    # Step 6: Return resume info
    return None if not resumed_model else resumed_model, initial_epoch, history


def _build_train_dataset(train_data, train_labels, config):
    """
    Builds the training dataset pipeline without one-hot encoding.

    Args:
        train_data (np.ndarray): Input training images.
        train_labels (np.ndarray): Corresponding class labels.
        config (Config): Configuration object.

    Returns:
        tf.data.Dataset: A prepared and prefetched dataset ready for training.
    """

    # Step 0: Print header for function execution
    print("\n🎯  _build_train_dataset is executing ...")

    # Step 1: Convert data arrays to tensors
    train_data_tensor = tf.convert_to_tensor(train_data)
    train_labels_tensor = tf.convert_to_tensor(train_labels)

    # Step 2: Create base dataset and shuffle
    dataset = tf.data.Dataset.from_tensor_slices((train_data_tensor, train_labels_tensor))
    dataset = dataset.shuffle(1024)

    # Step 3: No label transformation — pass raw integer labels
    # (One-hot encoding has been removed)

    # Step 4: Batch and prefetch
    dataset = dataset.batch(config.BATCH_SIZE)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


def _evaluate_validation(model, val_data, val_labels, config):
    """
    Evaluates the model on the validation set and returns loss and accuracy.

    Args:
        model (tf.keras.Model): Trained model to evaluate.
        val_data (np.ndarray): Validation images.
        val_labels (np.ndarray): Validation labels.
        config (Config): Configuration object for batch size.

    Returns:
        tuple: (val_loss, val_accuracy)
    """

    # Step 0: Print header for function execution
    print("\n🎯  _evaluate_validation is executing ...")

    # Step 1: Build tf.data dataset and batch
    val_dataset = tf.data.Dataset.from_tensor_slices((val_data, val_labels))
    val_dataset = val_dataset.batch(config.BATCH_SIZE)

    # Step 2: Initialize collections
    val_preds_all = []
    val_labels_all = []

    # Step 3: Run inference over batches
    for x_val, y_val in val_dataset:
        preds = model(x_val, training=False)
        val_preds_all.append(preds)
        val_labels_all.append(y_val)

    # Step 4: Concatenate predictions and labels
    val_preds = tf.concat(val_preds_all, axis=0)
    val_labels_cat = tf.concat(val_labels_all, axis=0)

    # Step 5: Compute loss and accuracy
    loss_fn = SparseCategoricalCrossentropy(from_logits=True)
    val_loss = loss_fn(val_labels_cat, val_preds).numpy()
    val_acc = tf.keras.metrics.sparse_categorical_accuracy(val_labels_cat, val_preds).numpy().mean()

    return val_loss, val_acc


def _prepare_callback(model, model_checkpoint_path: Path, config):
    """
    Creates a list of Keras callbacks:
    - Saves the best model based on validation accuracy
    - Saves a model after each epoch
    - Applies milestone-based StepLR decay (if SCHEDULE_MODE is enabled)
    - Applies optional learning rate warmup at start of training
    - Stops training early if validation stagnates (if EARLY_STOP_MODE is enabled)

    Args:
        model (tf.keras.Model): The model being trained.
        model_checkpoint_path (Path): Directory path for saving checkpoints.
        config (Config): Configuration object with callback settings.

    Returns:
        list: List of Keras callbacks to be passed to model.fit().
    """

    # Step 0: Print header for function execution
    print("\n🎯  _prepare_callback is executing ...")

    # Step 1: Define checkpoint file paths
    best_model_path = model_checkpoint_path / "best.keras"
    per_epoch_path = model_checkpoint_path / "epoch_{epoch:02d}.keras"

    # Step 2: Extract verbosity levels
    verbose_lr = config.SCHEDULE_MODE.get("verbose", 1)
    verbose_es = config.EARLY_STOP_MODE.get("verbose", 1)

    # Step 3: Initialize base callbacks — best model + per-epoch model saving
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

    # Step 4: Add EarlyStopping if enabled
    if config.EARLY_STOP_MODE.get("enabled", False):
        early_stop = EarlyStopping(
            monitor=config.EARLY_STOP_MODE.get("monitor", "val_accuracy"),
            patience=config.EARLY_STOP_MODE.get("patience", 5),
            restore_best_weights=config.EARLY_STOP_MODE.get("restore_best_weights", True),
            verbose=verbose_es
        )
        callbacks.append(early_stop)

    # Step 5: Add learning rate scheduler if enabled
    if config.SCHEDULE_MODE.get("enabled", False):
        scheduler_type = config.SCHEDULE_MODE.get("type", "").lower()

        # Step 5a: StepLR scheduler
        if scheduler_type == "step":
            class StepDecayScheduler(Callback):
                def __init__(self, optimizer, initial_lr, milestones, gamma):
                    super().__init__()
                    self.optimizer = optimizer
                    self.initial_lr = initial_lr
                    self.milestones = milestones
                    self.gamma = gamma

                def on_epoch_begin(self, epoch, logs=None):
                    lr = self.initial_lr
                    for milestone in self.milestones:
                        if epoch >= milestone:
                            lr *= self.gamma
                    self.model.optimizer.learning_rate.assign(lr)
                    print(f"\n🔁  StepLR applied — epoch {epoch}, learning rate set to {lr:.5f}\n")

            step_scheduler = StepDecayScheduler(
                optimizer=model.optimizer,
                initial_lr=config.OPTIMIZER.get("learning_rate", 0.1),
                milestones=config.SCHEDULE_MODE.get("milestones", [80, 120]),
                gamma=config.SCHEDULE_MODE.get("gamma", 0.1)
            )
            callbacks.append(step_scheduler)

        # Step 5b: ReduceLROnPlateau scheduler
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

        # Step 5c: Warmup scheduler (linear ramp-up for early epochs)
        if config.SCHEDULE_MODE.get("warmup_epochs", 0) > 0:
            class LinearWarmupScheduler(Callback):
                def __init__(self, optimizer, warmup_epochs, target_lr):
                    super().__init__()
                    self.optimizer = optimizer
                    self.warmup_epochs = warmup_epochs
                    self.target_lr = target_lr

                def on_epoch_begin(self, epoch, logs=None):
                    if epoch < self.warmup_epochs:
                        warmup_lr = self.target_lr * ((epoch + 1) / self.warmup_epochs)
                        self.model.optimizer.learning_rate.assign(warmup_lr)
                        print(f"\n🔥  WarmupLR applied — epoch {epoch}, learning rate set to {warmup_lr:.5f}\n")

            warmup_scheduler = LinearWarmupScheduler(
                optimizer=model.optimizer,
                warmup_epochs=config.SCHEDULE_MODE.get("warmup_epochs"),
                target_lr=config.OPTIMIZER.get("learning_rate", 0.1)
            )
            callbacks.append(warmup_scheduler)

    # Step 6: Return assembled callback list
    return callbacks


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

        # Step 0: Print header for constructor execution
        print("\n🎯  RecoveryCheckpoint.__init__")

        # Step 1: Initialize base Callback class
        super().__init__()

        # Step 2: Ensure checkpoint directory exists
        self.checkpoint_path = checkpoint_path
        self.checkpoint_path.mkdir(parents=True, exist_ok=True)

        # Step 3: Define target paths for model and training state
        self.model_path = checkpoint_path / "latest.keras"
        self.state_path = checkpoint_path / "state.json"

    def on_epoch_end(self, epoch, logs=None):
        """
        Callback method executed at the end of every training epoch.
        Saves the current model and the epoch number.

        Args:
            epoch (int): Current epoch number (0-based).
            logs (dict): Metrics from this epoch.
        """

        # Step 0: Print execution header
        print("\n🎯  on_epoch_end")

        # Step 1: Save model snapshot
        self.model.save(self.model_path)

        # Step 2: Record epoch number for resume
        with open(self.state_path, "w") as f:
            json.dump({"initial_epoch": epoch + 1}, f)

        # Step 3: Confirm checkpoint saved
        print(f"\n💾  Checkpointing experiment at epoch_{epoch + 1}\n")

        # Step 4: Log timestamp for freeze detection
        print(f"🕒  Recording time at {datetime.datetime.now(pytz.timezone('Asia/Tehran')).strftime('%H:%M')}\n")


def _run_training_loop(model, train_dataset, val_data, val_labels, config, model_checkpoint_path, initial_epoch):
    """
    Runs the core training loop with manual epoch control, validation, and best model saving.

    Args:
        model (tf.keras.Model): The model to train.
        train_dataset (tf.data.Dataset): Training dataset with all preprocessing applied.
        val_data (np.ndarray): Validation images.
        val_labels (np.ndarray): Validation labels.
        config (Config): Configuration object.
        model_checkpoint_path (Path): Directory to save best model.
        initial_epoch (int): Epoch to resume from.

    Returns:
        dict: History dictionary with loss and accuracy metrics.
    """

    # Step 0: Print header for function execution
    print("\n🎯  _run_training_loop is executing ...")

    # Step 1: Initialize tracking structures and metrics
    history = {"loss": [], "accuracy": [], "val_loss": [], "val_accuracy": []}
    best_acc = -1.0
    loss_fn = SparseCategoricalCrossentropy(from_logits=True)
    acc_fn = SparseCategoricalAccuracy()

    # Step 2: Define a compiled training step (tf.function for speed)
    @tf.function
    def train_step(x_batch, y_batch):
        with tf.GradientTape() as tape:
            preds = model(x_batch, training=True)
            loss = loss_fn(y_batch, preds)
        grads = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(grads, model.trainable_variables))
        acc_fn.update_state(tf.argmax(y_batch, axis=1), preds)
        return loss

    # Step 3: Loop over epochs
    for epoch in range(initial_epoch, config.EPOCHS_COUNT):
        print(f"\n📆  Epoch {epoch + 1}/{config.EPOCHS_COUNT}")
        epoch_loss = Mean()
        acc_fn.reset_state()

        for x_batch, y_batch in train_dataset:
            loss = train_step(x_batch, y_batch)
            epoch_loss.update_state(loss)

        epoch_acc = acc_fn.result().numpy()
        print(f"\n📊  Epoch {epoch + 1} — Loss: {epoch_loss.result():.4f} — Acc: {epoch_acc:.4f}")
        history["loss"].append(epoch_loss.result().numpy())
        history["accuracy"].append(epoch_acc)

        # Step 4: Evaluate on validation data
        val_loss_value, val_acc_value = _evaluate_validation(model, val_data, val_labels, config)
        history["val_loss"].append(val_loss_value)
        history["val_accuracy"].append(val_acc_value)

        print(f"\n📈  Val — Loss: {val_loss_value:.4f} — Acc: {val_acc_value:.4f}")

        # Step 5: Save best model
        if val_acc_value > best_acc:
            best_acc = val_acc_value
            model.save(model_checkpoint_path / "best.keras")
            print(f"\n💾  Saved new best model — Val Acc: {best_acc:.4f}")

    return history


def train_model(train_data, train_labels, model, model_number, run, config_name, config, val_data, val_labels):
    """
    Orchestrates the full training-or-resume cycle and returns:
        trained_model, history (SimpleNamespace), resumed_flag

    Args:
        train_data (np.ndarray): Training images (after split).
        train_labels (np.ndarray): Training labels (after split).
        model (tf.keras.Model): Compiled model instance.
        model_number (int): Model ID.
        run (int): Run number.
        config_name (str): Name of the config.
        config (Config): Config object.
        val_data (np.ndarray): Validation images.
        val_labels (np.ndarray): Validation labels.
    """


    # Step 0: Print function execution header
    print("\n🎯  train_model is executing ...")

    # Step 1: Define run_id and checkpoint path; ensure checkpoint directory exists
    run_id = f"m{model_number}_r{run}_{config_name}"
    ckpt_path = config.CHECKPOINT_PATH / run_id
    ckpt_path.mkdir(parents=True, exist_ok=True)

    # Step 2: Attempt to resume from checkpoint if available
    try:
        resumed_model, initial_epoch, history = _resume_from_checkpoint(run_id, config)
    except FileNotFoundError:
        resumed_model, initial_epoch, history = None, 0, None
    resumed = resumed_model is not None
    if resumed:
        model = resumed_model

    # Step 3: Build training dataset and prepare callbacks
    train_ds = _build_train_dataset(train_data, train_labels, config)
    callbacks = _prepare_callback(model, ckpt_path, config)
    callbacks.append(RecoveryCheckpoint(ckpt_path))

    # Step 4: Optionally log learning rate after each batch
    class LearningRateLogger(Callback):
        def on_train_batch_end(self, batch, logs=None):
            logs = logs or {}
            logs["lr"] = float(self.model.optimizer.learning_rate)
    callbacks.append(LearningRateLogger())

    # Step 5: Train with Keras model.fit (handles epochs, validation, callbacks)
    history_obj = model.fit(
        train_ds,
        validation_data=(val_data, val_labels),
        epochs=config.EPOCHS_COUNT,
        initial_epoch=initial_epoch,
        callbacks=callbacks,
        verbose=2
    )

    # Step 6: Merge any old history if training was resumed
    history_dict = history_obj.history
    history_dict = _merge_old_history(ckpt_path, history_dict, initial_epoch)
    history = SimpleNamespace(history=history_dict)

    # Step 7: Finalize training by saving model and history
    _finalize_training(
        model, history_dict,
        model_number, run, config_name,
        config, ckpt_path
    )

    # Step 8: Return final model, history namespace, and resume status
    return model, history, resumed


def _merge_old_history(model_checkpoint_path, new_history, initial_epoch):
    """
    Merges old training history from disk with new training history if resuming.

    Args:
        model_checkpoint_path (Path): Path to checkpoint folder containing history.json.
        new_history (dict): Current training history collected in this run.
        initial_epoch (int): Epoch index indicating if training resumed.

    Returns:
        dict: Combined training history.
    """

    # Step 0: Print header for function execution
    print("\n🎯  _merge_old_history is executing ...")

    # Step 1: Only merge if training resumed from a non-zero epoch
    if initial_epoch == 0:
        return new_history

    # Step 2: Define path to history file
    history_file = model_checkpoint_path / "history.json"

    # Step 3: Check if history file exists
    if history_file.exists():
        try:
            with open(history_file, "r") as f:
                old_history = json.load(f)

            # Step 4: Merge history keys
            for key in new_history:
                new_history[key] = old_history.get(key, []) + new_history[key]

        except Exception as e:
            print(f"\n⚠️  Failed to merge old history:\n→ {e}")

    return new_history


def _finalize_training(model, history, model_number, run, config_name, config, model_checkpoint_path):
    """
    Saves the final model and training history to disk after training completes.

    Args:
        model (tf.keras.Model): Trained model.
        history (dict): Final training history.
        model_number (int): ID of the model.
        run (int): Run index.
        config_name (str): Name of the active config.
        config (Config): Configuration object with paths.
        model_checkpoint_path (Path): Path where history should be saved.

    Returns:
        None
    """

    # Step 0: Print header for function execution
    print("\n🎯  _finalize_training is executing ...")

    # Step 1: Save the trained model
    model_path = config.MODEL_PATH / f"m{model_number}_r{run}_{config_name}.keras"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(model_path)
    print(f"\n💾  Final model saved:\n→ {model_path}")

    # Step 2: Save the full training history
    try:
        class DummyHistory: pass
        h = DummyHistory()
        h.history = history
        save_training_history(model_checkpoint_path / "history.json", h)
        print(f"\n📊  Training history saved:\n→ {model_checkpoint_path / 'history.json'}")
    except Exception as e:
        print(f"\n⚠️   Failed to save training history:\n→ {e}")


# Print confirmation message
print("\n✅  train.py successfully executed")
