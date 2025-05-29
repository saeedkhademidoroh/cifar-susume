# Import standard libraries
import hashlib
import json
import os
import shutil
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Import third-party libraries
from keras.api.models import load_model
import numpy as np
import tensorflow as tf
from tensorflow.python.client import device_lib

# Import project-specific libraries
from config import CONFIG


def clean_old_output(flag=False):
    """
    Cleans model, checkpoint, and error directories if CLEAN_MODE is enabled.

    Primarily used to remove previous run artifacts before starting a new experiment.

    Args:
        flag (bool): If True, triggers cleanup. Otherwise, does nothing.

    Returns:
        None
    """

    # Step 0: Print header for function execution
    print("\n🎯  clean_old_output is executing ...")

    # Step 1: Check cleanup flag
    if flag:

        # Step 2: Announce cleanup initiation
        print("\n🧼  CLEAN_MODE is ON - Cleaning old output directories ...")

        # Step 3: Define target directories for cleanup
        targets = [
            CONFIG.MODEL_PATH,
            CONFIG.ERROR_PATH,
            CONFIG.CHECKPOINT_PATH,
        ]

        # Step 4: Verify paths are all valid Path objects
        if not all(isinstance(p, Path) for p in targets):
            raise TypeError(
                "\n❌  Error from utility.py at clean_old_output()!\n"
                "→ One or more CONFIG paths are not valid pathlib.Path objects.\n"
            )

        # Step 5: Remove each target directory if it exists
        print(f"\n🧹  Cleaning old experiment output directories:")
        for path in targets:
            if path.exists():
                print(f"→ {path}  (deleted)")
                shutil.rmtree(path, ignore_errors=True)
            else:
                print(f"→ {path}  (not found)")

    else:
        # Step 6: Cleanup is disabled — notify and exit
        print("\n🧼  CLEAN_MODE is OFF — skipping old output directories ...")


def ensure_output_directories(config):
    """
    Ensures existence of all required output folders for logging, checkpoints,
    results, models, and error traces. This prevents runtime failures due to
    missing directories during experiment execution.

    Args:
        config (Config): Configuration object containing path definitions.

    Returns:
        None
    """

    # Step 0: Print header for function execution
    print("\n🎯  ensure_output_directories is executing ...")

    # Step 1: Announce directory setup action
    print(f"\n📂  Ensuring output directories")

    # Step 2: Gather required output paths
    targets = [
        config.LOG_PATH,
        config.CHECKPOINT_PATH,
        config.RESULT_PATH,
        config.MODEL_PATH,
        config.ERROR_PATH
    ]

    # Step 3: Validate that all targets are Path instances
    if not all(isinstance(p, Path) for p in targets):
        raise TypeError(
            "\n❌  Error from utility.py at ensure_output_directories()!\n"
            "→ One or more CONFIG paths are not valid pathlib.Path objects.\n"
        )

    # Step 4: Create each required directory if it doesn't exist
    for path in targets:
        path.mkdir(parents=True, exist_ok=True)
        print(f"→ {path}")  # Print confirmed path


def extract_history_metrics(history):
    """
    Function to extract min/max training and validation metrics from history.

    Handles both `History` objects and plain dictionaries. Computes:
    - Minimum training loss and corresponding epoch
    - Maximum training accuracy and corresponding epoch
    - (Optional) Minimum validation loss and maximum validation accuracy with epochs

    Args:
        history (History or dict): Training history object or dictionary

    Returns:
        dict: Dictionary containing key metrics and their epochs
    """

    # Step 0: Print header for function execution
    print("\n🎯  extract_history_metrics is executing ...")

    # Step 1: Validate input format — must be dict or object with `.history`
    if not isinstance(history, (dict, object)) or not (hasattr(history, "history") or isinstance(history, dict)):
        raise TypeError(
            "\n\n❌  Error from utility.py at extract_history_metrics()!\n"
            "→ Invalid history format — must be dict or object with .history attribute\n\n"
        )

    # Step 2: If it's a History object, extract its `.history` dictionary
    history = history.history if hasattr(history, "history") else history

    # Step 3: Retrieve core training metric lists from history dictionary
    train_loss = history.get("loss", [])
    train_acc = history.get("accuracy", [])

    # Step 4: Fail early if either training loss or accuracy are missing
    if not train_loss or not train_acc:
        raise ValueError(
            "\n\n❌  Error from utility.py at extract_history_metrics()!\n"
            "→ Training history is incomplete — loss/accuracy may be missing\n\n"
        )

    # Step 5: Extract training statistics — min loss and max accuracy with epochs
    metrics = {
        "min_train_loss": min(train_loss),
        "min_train_loss_epoch": train_loss.index(min(train_loss)) + 1,
        "max_train_acc": max(train_acc),
        "max_train_acc_epoch": train_acc.index(max(train_acc)) + 1,
    }

    # Step 6: Retrieve optional validation metrics
    val_loss = history.get("val_loss", [])
    val_acc = history.get("val_accuracy", [])

    # Step 7: Warn if validation metrics are missing (may be due to LIGHT_MODE)
    if not val_loss or not val_acc:
        print(
            "\n⚠️  Warning from utility.py at extract_history_metrics()!"
            "\n→ Validation metrics are missing — possibly due to LIGHT_MODE.\n"
        )

    # Step 8: Append validation loss metrics if available
    if val_loss:
        metrics["min_val_loss"] = min(val_loss)
        metrics["min_val_loss_epoch"] = val_loss.index(min(val_loss)) + 1

    # Step 9: Append validation accuracy metrics if available
    if val_acc:
        metrics["max_val_acc"] = max(val_acc)
        metrics["max_val_acc_epoch"] = val_acc.index(max(val_acc)) + 1

    # Step 10: Print extracted metrics for traceability
    print(f"\n📈  Training metrics have been extracted:")
    print(f"→ Min train loss: {metrics['min_train_loss']} at epoch {metrics['min_train_loss_epoch']}")
    print(f"→ Max train acc:  {metrics['max_train_acc']} at epoch {metrics['max_train_acc_epoch']}")
    if val_loss:
        print(f"→ Min val loss:  {metrics['min_val_loss']} at epoch {metrics['min_val_loss_epoch']}")
    if val_acc:
        print(f"→ Max val acc:   {metrics['max_val_acc']} at epoch {metrics['max_val_acc_epoch']}")

    # Step 11: Return collected metrics
    return metrics


def recover_training_history(config, run_id):
    """
    Attempts to load saved training history from disk using the run ID.

    Used when resuming an experiment and the in-memory `history` is unavailable.
    Searches for a `history.json` file under the appropriate checkpoint folder.

    Args:
        config (Config): Configuration object containing path definitions.
        run_id (str): Unique identifier for the run (e.g., "m9_r1_default").

    Returns:
        object: A dummy object containing `.history` if recovery succeeds.

    Raises:
        FileNotFoundError: If the history file is not found.
        ValueError: If history file is found but malformed or unreadable.
    """

    # Step 0: Print header for function execution
    print("\n🎯  recover_training_history is executing ...")

    # Step 1: Construct full path to the expected history.json file
    history_file = config.CHECKPOINT_PATH / run_id / "history.json"
    print(f"\n📄  Attempting to recover history from:\n→ {history_file}")

    # Step 2: Check if the file exists before attempting to load
    if history_file.exists():
        try:
            # Step 3: Open and parse the history JSON
            with open(history_file, "r") as f:
                history_data = json.load(f)

            # Step 4: Validate that the loaded content is a dictionary
            if not isinstance(history_data, dict):
                raise TypeError(
                    "\n\n❌  Error from utility.py at recover_training_history()!\n"
                    "→ Loaded history is not a dictionary.\n\n"
                )

            # Step 5: Wrap dictionary in a dummy object with `.history` attribute
            class DummyHistory:
                pass

            h = DummyHistory()
            h.history = history_data

            # Step 6: Confirm successful recovery
            print("\n✅  Training history successfully recovered.")
            return h

        except Exception as e:
            # Step 7: Raise parsing failure as ValueError with context
            raise ValueError(
                f"\n\n❌  Error from utility.py at recover_training_history()!\n"
                f"→ Failed to parse training history JSON:\n→ {history_file}\n→ {e.__class__.__name__}: {e}\n\n"
            )

    # Step 8: Raise error if history file does not exist
    raise FileNotFoundError(
        f"\n\n❌  Error from utility.py at recover_training_history()!\n"
        f"→ History file not found:\n→ {history_file}\n\n"
    )


def load_from_checkpoint(model_checkpoint_path: Path):
    """
    Attempts to resume training by loading the latest saved model and training state.

    Args:
        model_checkpoint_path (Path): Directory containing checkpoint files.

    Returns:
        tuple: (model, initial_epoch)
    """

    # Step 0: Print header for function execution
    print("\n🎯  load_from_checkpoint is executing ...")

    # Step 1: Define checkpoint paths
    state_path = model_checkpoint_path / "state.json"
    model_path = model_checkpoint_path / "latest.keras"

    # Step 2: Check if both files exist and load them
    if model_path.exists() and state_path.exists():
        print(f"\n📦  Loading model checkpoint from:\n→ {model_path}")
        with open(state_path, "r") as f:
            state = json.load(f)
        model = load_model(model_path)
        return model, state.get("initial_epoch", 0)

    # Step 3: Raise error if checkpoint is missing
    raise FileNotFoundError(
        f"\n\n❌  Error from utility.py at load_from_checkpoint()!\n"
        f"→ Missing checkpoint components:\n→ {model_path}\n→ {state_path}\n\n"
    )


def save_training_history(history_file: Path, history_obj):
    """
    Saves a Keras-style training history object to a JSON file.

    Args:
        history_file (Path): File path to save history.
        history_obj: Object with a `.history` attribute.
    """

    # Step 0: Print header for function execution
    print("\n🎯  save_training_history is executing ...")

    # Step 1: Write history dictionary to disk
    try:
        with open(history_file, "w") as f:
            json.dump(history_obj.history, f)
        print(f"\n💾  Training history saved to:\n→ {history_file}")
    except Exception as e:
        raise IOError(
            f"\n\n❌  Error from utility.py at save_training_history()!\n"
            f"→ Failed to write training history JSON:\n→ {history_file}\n→ {e.__class__.__name__}: {e}\n\n"
        )


def split_dataset(train_data, train_labels, light_mode):
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

    # Step 0: Print header for function execution
    print("\n🎯  split_dataset is executing ...")

    # Step 1: Validate dataset size before splitting
    if len(train_data) < 10:
        raise ValueError(
            f"\n\n❌  Error from utility.py at split_dataset()!\n"
            f"→ Dataset too small to split — fewer than 10 samples available.\n→ ValueError\n\n"
        )


    # Step 2: Split based on light_mode flag
    if light_mode:
        # Step 2a: Use 20% of the data for validation
        val_split = int(0.2 * len(train_data))
        val_data = train_data[-val_split:]
        val_labels = train_labels[-val_split:]
        train_data = train_data[:-val_split]
        train_labels = train_labels[:-val_split]
    else:
        # Step 2b: Use fixed last 5000 samples for validation
        val_data = train_data[-5000:]
        val_labels = train_labels[-5000:]
        train_data = train_data[:-5000]
        train_labels = train_labels[:-5000]

    # Step 3: Return split subsets
    return train_data, train_labels, val_data, val_labels


def print_training_context(config, run_id=None):
    """
    Prints environment, compute, and configuration details to help trace and reproduce experiments.

    Args:
        config (Config): Loaded configuration object.
        run_id (str, optional): Unique identifier for the training run.
    """

    # Step 0: Print header for function execution
    print("\n🎯  print_training_context is executing ...")

    # Step 1: Print time, run ID, Python and library versions
    print(f"\n⏰  Start Time: {datetime.now(timezone(timedelta(hours=3, minutes=30))).strftime('%Y-%m-%d %H:%M:%S')}")
    if run_id:
        print(f"🆔  Run ID:            {run_id}")
    print(f"🐍  Python Version:    {sys.version.split()[0]}")
    print(f"📦  TensorFlow:        {tf.__version__}")
    print(f"📦  NumPy:             {np.__version__}")

    # Step 2: Print available compute devices and memory info
    print("\n🖥️   Available Devices:")
    devices = device_lib.list_local_devices()
    for device in devices:
        print(f"  • {device.name} ({device.device_type})")
        if hasattr(device, "memory_limit"):
            mem_mb = int(device.memory_limit) // (1024 * 1024)
            print(f"    ⮡ Memory Limit: {mem_mb} MB")
    print(f"→ Total devices:     {len(devices)}")

    # Step 3: Print GPU detection and memory info
    gpus = tf.config.list_physical_devices("GPU")
    print(f"\n🧮  GPU Detected:     {'YES' if gpus else 'NO'}")
    for gpu in gpus:
        details = tf.config.experimental.get_device_details(gpu)
        mem = details.get("device_memory", None)
        mem_str = f" — {mem // (1024 * 1024)} MB" if mem else ""
        print(f"  • {gpu.name}{mem_str}")

    # Step 4: Print config signature from live config object
    try:
        config_str = json.dumps(config.__dict__, sort_keys=True, default=str)
        config_hash = hash(config_str)
        print(f"\n🔏  Live Config Signature: {config_hash}")
    except Exception:
        print("🔏  Live Config Signature: (unavailable)")

    # Step 5: Print loaded config file hash (static trace of config.json)
    try:
        with open(CONFIG.CONFIG_PATH, "rb") as f:
            content = f.read()
            hash_digest = hashlib.md5(content).hexdigest()
        print(f"📁  Config File Hash:      {hash_digest} — ({CONFIG.CONFIG_PATH.name})")
    except Exception as e:
        print(f"⚠️  Unable to compute config hash:\n→ {e.__class__.__name__}: {e}")

    # Step 6: Print core training hyperparameters
    print(f"\n🧪  Training Settings:")
    print(f"→ Epochs:            {config.EPOCHS_COUNT}")
    print(f"→ Batch Size:        {config.BATCH_SIZE}")
    print(f"→ Optimizer:         {config.OPTIMIZER['type'].upper()} (lr = {config.OPTIMIZER['learning_rate']})")
    print(f"→ Momentum:          {config.OPTIMIZER.get('momentum', 0.0)}")
    print(f"→ L2 Regularization: {'ON' if config.L2_MODE['enabled'] else 'OFF'} (λ = {config.L2_MODE['lambda']})")
    print(f"→ Dropout:           {'ON' if config.DROPOUT_MODE['enabled'] else 'OFF'} (rate = {config.DROPOUT_MODE['rate']})")

    # Step 7: Print augmentation config
    print(f"\n🧬  Augmentation:")
    print(f"→ Augmentation:      {'ON' if config.AUGMENT_MODE['enabled'] else 'OFF'}", end="")
    if config.AUGMENT_MODE['enabled']:
        print(" —", end=" ")
        flags = []
        if config.AUGMENT_MODE.get("random_crop", False): flags.append("Random Crop")
        if config.AUGMENT_MODE.get("random_flip", False): flags.append("Horizontal Flip")
        if config.AUGMENT_MODE.get("cutout", False): flags.append("Cutout")
        if config.AUGMENT_MODE.get("color_jitter", False): flags.append("Color Jitter")
        print(", ".join(flags))
    else:
        print()

    # Step 8: Print special training modes
    print(f"\n🔧  Mode Flags:")

    print(f"→ Weight Averaging:  {'ON' if config.AVERAGE_MODE['enabled'] else 'OFF'}", end="")
    if config.AVERAGE_MODE['enabled']:
        print(f" — from epoch {config.AVERAGE_MODE.get('start_epoch', '?')}")
    else:
        print()

    print(f"→ TTA:               {'ON' if config.TTA_MODE['enabled'] else 'OFF'}", end="")
    if config.TTA_MODE['enabled']:
        print(f" — {config.TTA_MODE.get('runs', 1)} passes/sample")
    else:
        print()

    print(f"→ Early Stopping:    {'ON' if config.EARLY_STOP_MODE['enabled'] else 'OFF'}", end="")
    if config.EARLY_STOP_MODE['enabled']:
        print(f" — patience {config.EARLY_STOP_MODE.get('patience', '?')}, restore: {config.EARLY_STOP_MODE.get('restore_best_weights', False)}")
    else:
        print()

    print(f"→ LR Scheduler:      {'ON' if config.SCHEDULE_MODE['enabled'] else 'OFF'}", end="")
    if config.SCHEDULE_MODE['enabled']:
        print(f" — warmup {config.SCHEDULE_MODE.get('warmup_epochs', 0)} epochs, γ = {config.SCHEDULE_MODE.get('gamma', '?')}")
    else:
        print()


class Tee:
    """
    Custom class to duplicate standard output/error to multiple streams,
    such as console and log file simultaneously.
    """

    def __init__(self, *streams):
        """
        Step 0: Initialize the Tee instance with one or more writable stream objects.

        Args:
            *streams: Arbitrary number of writable stream targets (e.g., sys.stdout, file handles).
        """

        print("\n🎯  Tee.__init__ is executing ...")
        self.streams = streams

    def write(self, data):
        """
        Step 1: Write incoming data to all attached output streams.

        Args:
            data (str): The string to be written to all streams.
        """

        for s in self.streams:
            try:
                s.write(data)
                s.flush()
            except Exception as e:
                print("\n\n❌  Error from utility.py at Tee.write()!\n""→ Failed to write to stream.\n"f"→ {e}\n\n")

    def flush(self):
        """
        Step 2: Flush all attached streams to ensure buffered data is written out.
        """

        for s in self.streams:
            try:
                s.flush()
            except Exception as e:
                print("\n\n❌  Error from utility.py at Tee.flush()!\n""→ Failed to flush stream.\n"f"→ {e}\n\n")


def initialize_logging(timestamp):
    """
    Initializes logging to both console and a log file.

    Creates timestamped log and result files, and redirects stdout/stderr
    to also write to the log file using the Tee class.

    Args:
        timestamp (str): Timestamp string to uniquely name log and result files.

    Returns:
        tuple: (log_stream, result_file_path, empty_result_list)
    """

    # Step 0: Print header for function execution
    print("\n🎯  initialize_logging is executing ...")

    # Step 1: Ensure log directory exists
    CONFIG.LOG_PATH.mkdir(parents=True, exist_ok=True)

    # Step 2: Create and open the log file for writing
    log_file = CONFIG.LOG_PATH / f"log_{timestamp}.txt"
    try:
        log_stream = open(log_file, "a", buffering=1)  # line-buffered for real-time logging
    except Exception as e:
        raise OSError(
            f"\n\n❌  Error from utility.py at initialize_logging()!\n"
            f"→ Failed to open log file:\n→ {log_file}\n→ {e}\n\n"
        )

    # Step 3: Redirect stdout and stderr to both terminal and log file
    sys.stdout = Tee(sys.__stdout__, log_stream)
    sys.stderr = Tee(sys.__stderr__, log_stream)

    # Step 4: Confirm log setup to user and log file
    print(f"\n📜  Logging experiment output:\n→ {log_file.resolve()}", flush=True)

    # Step 5: Log environment info at the very start for traceability
    print(f"\n🧾  Runtime environment:")
    print(f"→ Python:         {sys.version.split()[0]}")
    print(f"→ TensorFlow:     {tf.__version__}")
    print(f"→ NumPy:          {np.__version__}")
    print(f"→ Log file:       {log_file.name}")
    print(f"→ Timestamp:      {timestamp}")

    # Step 6: Ensure result directory exists and define result file path
    CONFIG.RESULT_PATH.mkdir(parents=True, exist_ok=True)
    result_file = CONFIG.RESULT_PATH / f"result_{timestamp}.json"

    # Step 7: Return handles and placeholder result list
    return log_stream, result_file, []


def load_previous_results(result_file, all_results):
    """
    Loads previously recorded experiment results to prevent redundant runs.

    Parses an existing result JSON file (if available) and extracts:
    - Completed (model, run, config) tuples for deduplication
    - Existing result entries, appended to the provided list

    This function ensures that experiments already completed are not repeated,
    by maintaining a memory of all prior combinations.

    Args:
        result_file (Path): Path to the result_<timestamp>.json file.
        all_results (list): A mutable list to be populated with past results.

    Returns:
        set: A set of (model_number, run_id, config_name) tuples already recorded.
    """

    # Step 0: Print header for function execution
    print("\n🎯  load_previous_results is executing ...")

    # Step 1: Check if the result file for the current session exists
    if result_file.exists(): #
        print(f"📄  Found existing result file: {result_file}. Attempting to load.")
        try:
            # Step 2: Attempt to open and load the result file
            with open(result_file, "r") as jf:
                existing_results_for_session = json.load(jf)

            # Step 3: Append these existing results to the main all_results list
            all_results.extend(existing_results_for_session)
            print(f"✅  Loaded {len(existing_results_for_session)} results from existing session file.")

            # Step 4: Extract (model, run, config_name) tuples for deduplication
            # Using "config_name" as this is what _create_evaluation_dictionary saves.
            completed_triplets = set()
            for entry in existing_results_for_session:
                if "model" in entry and "run" in entry and "config_name" in entry:
                    completed_triplets.add(
                        (entry["model"], entry["run"], entry["config_name"])
                    )
                else:
                    # Step 4a: Warn about malformed entries if any
                    print(f"⚠️  Skipping malformed entry in {result_file}: {entry}")
            return completed_triplets

        # Step 5: Handle error if loading/parsing fails
        except Exception as e:
            print(f"⚠️  Warning: Failed to parse existing result file {result_file}. Proceeding as if no prior results for this session. Error: {e}")
            # Step 5a: Reset all_results if parsing failed to prevent partial loads
            all_results.clear()
            return set()
    else:
        # Step 6: If result file does not exist, it's a fresh run for this timestamp.
        print(f"\nℹ️  Result file not found!\n→ {result_file}")
        print("→ Assuming fresh run for this session ...")
        return set() # Return empty set if file doesn't exist


def log_to_json(path, key, record=None, error=False):
    """
    Logs structured data to a JSON file for experiment tracking.

    Supports two modes:
    - Result mode: Appends a new record under a specified key in result.json
    - Error mode: Saves a standalone timestamped error JSON file

    Automatically adds UTC timestamps to each record if not provided.

    Args:
        path (Path or str): Directory where the log file will be saved.
        key (str): Key under which to store the record (ignored if error=True).
        record (dict, optional): Data dictionary to log. Defaults to empty dict.
        error (bool): If True, logs as error_<timestamp>.json instead of appending to result.json.

    Returns:
        None
    """

    # Step 0: Print header for function execution
    print("\n🎯  log_to_json is executing ...")

    # Step 1: Default to empty record if not provided
    if record is None:
        record = {}

    # Step 2: Ensure output directory exists
    os.makedirs(path, exist_ok=True)

    # Step 3: Append timestamps if missing
    if "timestamp" not in record:
        ts = time.time()
        record["timestamp"] = ts  # raw timestamp
        record["timestamp_utc"] = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()

        tehran_offset = timezone(timedelta(hours=3, minutes=30))
        record["timestamp_tehran"] = datetime.fromtimestamp(ts, tz=tehran_offset).isoformat()

    # Step 4: Handle error logging mode
    if error:
        try:
            error_file = Path(path) / f"error_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json"
            with open(error_file, "w") as f:
                json.dump(record, f, indent=2)

            print(f"\n📛  Error log saved to:\n→ {error_file}")

        except Exception as e:
            raise IOError(
                f"\n\n❌  Error from utility.py at log_to_json()!\n"
                f"→ Failed to write error log:\n→ {e.__class__.__name__}: {e}\n\n"
            )
        return


    # Step 5: Handle normal logging mode
    log_file = Path(path) / "result.json"

    try:
        # Step 5a: Load existing log or initialize
        if log_file.exists():
            with open(log_file, "r") as f:
                data = json.load(f)
        else:
            data = {}

        # Step 5b: Initialize key if missing
        if key not in data:
            data[key] = []

        # Step 5c: Append record and write back
        data[key].append(record)
        with open(log_file, "w") as f:
            json.dump(data, f, indent=2)

        print(
            f"\n📝  Logging experiment result ...\n"
            f"→ Key: '{key}'\n→ File: '{log_file.name}'",
            flush=True  # Ensures immediate print in redirected streams
        )

    except Exception as e:
        raise IOError(
            f"\n\n❌  Error from utility.py at log_to_json()!\n"
            f"→ Failed to write result log:\n→ {log_file}\n→ {e}\n\n"
        )


def to_json_compatible(obj):
    """
    Recursively converts an object to a format compatible with JSON serialization.

    Handles nested structures (dicts/lists), NumPy scalars, and TensorFlow tensors.

    Args:
        obj (any): The input object (may be dict, list, NumPy scalar, tf.Tensor, etc.)

    Returns:
        any: JSON-serializable equivalent of the input object
    """

    # Step 0: Print function execution header (only top-level)
    if "__json_trace_triggered__" not in globals():
        print("\n🎯  to_json_compatible is executing ...")
        globals()["__json_trace_triggered__"] = True

    # Step 1: Convert dictionary by processing all values recursively
    if isinstance(obj, dict):
        return {k: to_json_compatible(v) for k, v in obj.items()}

    # Step 2: Convert list by processing each element recursively
    elif isinstance(obj, list):
        return [to_json_compatible(v) for v in obj]

    # Step 3: Convert NumPy scalar types (e.g., np.float32) to native Python types
    elif isinstance(obj, (np.generic,)):
        return obj.item()

    # Step 4: Convert TensorFlow tensors
    elif isinstance(obj, tf.Tensor):
        return obj.numpy().item() if tf.rank(obj) == 0 else obj.numpy().tolist()

    # Step 5: Allow primitive JSON-safe types as-is
    elif isinstance(obj, (int, float, str, bool)) or obj is None:
        return obj

    # Step 6: Raise error for unsupported types
    else:
        raise TypeError(
            f"\n\n❌  Error from utility.py at to_json_compatible()!\n"
            f"→ Unsupported type for JSON serialization: {type(obj)}\n\n"
        )


# Print module successfully executed
print("\n✅  utility.py successfully executed.")
