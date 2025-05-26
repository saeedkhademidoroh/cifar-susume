# Import standard libraries
import datetime
import json
import sys
import time
import traceback

# Import third-party libraries
import numpy as np
import tensorflow as tf
from keras.api.models import load_model

# Import project-specific libraries
from config import CONFIG
from data import build_dataset
from evaluate import evaluate_model, extract_history_metrics
from log import log_to_json
from model import build_model
from train import train_model


# Function to recursively convert NumPy/TensorFlow types to native Python types
def _to_json_compatible(obj):
    # If the object is a dictionary, convert all values recursively
    if isinstance(obj, dict):
        return {k: _to_json_compatible(v) for k, v in obj.items()}

    # If the object is a list, convert all elements recursively
    elif isinstance(obj, list):
        return [_to_json_compatible(v) for v in obj]

    # Convert NumPy scalar types to native Python types
    elif isinstance(obj, (np.generic,)):
        return obj.item()

    # Convert TensorFlow scalar tensors
    elif isinstance(obj, tf.Tensor):
        if tf.rank(obj) == 0:
            return obj.numpy().item()
        else:
            return obj.numpy().tolist()

    # Already compatible types are returned as-is
    return obj


# Function to run all experiments in pipeline
def run_pipeline(pipeline):
    """
    Function to run multiple experiments defined as (model_number, config_name) tuples.

    Each entry is processed sequentially, with duplicate runs automatically skipped
    if results are already logged. The function handles:

    - Configuration loading
    - Directory setup
    - Logging initialization
    - Per-run metadata tracking
    - Result persistence

    Args:
        pipeline (list of tuple): List of (model_number: int, config_name: str)

    Returns:
        None
    """

    # Print header for function execution
    print("\n🎯  run_pipeline")

    # Generate timestamp for consistent filenames
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Initialize logging and result tracking
    log_stream, result_file, all_results = _initialize_logging(timestamp)

    try:
        # Load previously completed (model, config) combinations
        completed_triplets = _load_previous_results(result_file, all_results)

        # Dictionary to track how many times each model_number has been run
        model_run_counter = {}

        # Iterate through each experiment and assign unique run ID
        for i, (model_number, config_name) in enumerate(pipeline):
            print(f"\n⚙️   Piplining experiment {i+1}/{len(pipeline)}")

            # Build full path to the selected config file
            config_path = CONFIG.CONFIG_PATH / f"{config_name}.json"

            # Increment run count for this model_number
            model_run_counter[model_number] = model_run_counter.get(model_number, 0) + 1
            run = model_run_counter[model_number]

            # Execute one experiment with specified parameters
            _run_single_pipeline_entry(
                model_number=model_number,
                config_path=config_path,
                config_name=config_name,
                run=run,
                timestamp=timestamp,
                completed_triplets=completed_triplets,
                all_results=all_results,
                result_file=result_file
            )


        # Print final summary of all completed experiments
        print(f"\n📦   Completed {len(all_results)} total experiment runs")

        # Save all accumulated results after pipeline execution
        with open(result_file, "w") as jf:
            json.dump(_to_json_compatible(all_results), jf, indent=2)

    finally:
        # Restore standard output/error streams and close log
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        if log_stream:
            log_stream.close()


# Function to run a single pipeline entry
def _run_single_pipeline_entry(model_number, config_path, config_name, run, timestamp, completed_triplets, all_results, result_file):
    """
    Function to execute one training-evaluation cycle for a specific model and config.

    Handles:
    - Config loading and output path setup
    - Dataset loading per model variant
    - Model instantiation and training
    - Resuming logic and history recovery
    - Final evaluation and metric logging
    - Result storage in structured format

    Args:
        model_number (int): Model variant to build
        config_path (Path): Path to the config JSON
        config_name (str): Name of the config used
        run (int): Index of current run for this model
        timestamp (str): Pipeline-level timestamp string
        completed_triplets (set): Previously logged (model, run, config_name) tuples
        all_results (list): In-memory list of result entries
        result_file (Path): Output file to save JSON results

    Returns:
        Path: The updated result_file path after logging
    """

    # Print header for function execution
    print("\n🎯  _run_single_pipeline_entry")

    # Load dynamic configuration for this run
    config = CONFIG.load_config(config_path)
    assert isinstance(config, CONFIG.__class__), "config must be a Config instance"

    # Create output folders if missing
    _ensure_output_directories(config)

    # Skip if already completed
    if (model_number, run, config_name) in completed_triplets:
        print(f"\n⏩  Skipping experiment m{model_number}_r{run} with '{config_name}'")
        return result_file  # Return early since result already exists

    # Announce launch
    print(f"\n🚀  Launching experiment m{model_number}_r{run} with '{config_name}'")
    start_time = time.time()

    # Construct unique identifier for this run — used in paths, logs, and checkpoints
    run_id = f"m{model_number}_r{run}_{config_name}"

    try:
        # Load dataset for this model variant
        train_data, train_labels, test_data, test_labels = build_dataset(config)

        # Build model architecture
        model = build_model(model_number, config)

        # Train model (resumable)
        trained_model, history, resumed = train_model(
            train_data, train_labels, model, model_number, run, config_name, config
        )

        # Recover history if training was resumed and history is missing
        if resumed and (history is None or not hasattr(history, "history")):
            history = _recover_training_history(config, run_id)

        # Extract training/validation metrics
        metrics = extract_history_metrics(history)

        # Load best-performing model before final evaluation
        best_model_path = config.CHECKPOINT_PATH / run_id / "best.keras"
        trained_model = load_model(best_model_path)
        print(f"\n📥  Restored best model from:\n{best_model_path}")

        # Evaluate best model on test data
        metrics = evaluate_model(
            trained_model, history, test_data, test_labels, config
        )
        final_test_loss = metrics["final_test_loss"]
        final_test_acc = metrics["final_test_acc"]


        # Build evaluation dictionary for logging
        evaluation = _create_evaluation_dictionary(
            model_number, run, config_name,
            time.time() - start_time, config,
            metrics, final_test_loss, final_test_acc
        )

        # Print summary to console
        print("\n📊  Dumping experiment results:")

        # Safely print evaluation dictionary as formatted JSON
        print(json.dumps([_to_json_compatible(evaluation)], indent=2))

        # Append to in-memory result list and save to disk
        all_results.append(evaluation)
        with open(result_file, "w") as jf:
            json.dump(_to_json_compatible(all_results), jf, indent=2)

        print(f"\n✅   m{model_number} run {run} with '{config_name}' successfully executed")

    except Exception as e:
        # On failure, log error details to error file
        log_to_json(
            config.ERROR_PATH, key=run_id,
            record={
                "model": model_number,
                "run": run,
                "config_name": config_name,
                "error": str(e),
                "exception_type": type(e).__name__,
                "trace": traceback.format_exc()
            },
            error=True
        )
        raise  # Re-raise the exception for upstream handling

    return result_file  # Return final result file path after logging


# Function to load previous results
def _load_previous_results(result_file, all_results):
    """
    Loads existing results from a previous result JSON file and updates the in-memory list.

    Returns a set of (model, run, config) identifiers already logged, to prevent duplicates.

    Args:
        result_file (Path): Path to existing result JSON file.
        all_results (list): The current in-memory result log (will be extended in-place).

    Returns:
        set: Set of (model, run, config) tuples to use for deduplication.
    """

    # Print header for function execution
    print("\n🎯  _load_previous_results")

    if result_file.exists():
        with open(result_file, "r") as jf:
            existing = json.load(jf)
            all_results.extend(existing)

            # Return set of identifiers to skip duplicates
            return {
                (entry["model"], entry.get("run", 1), entry.get("config", "default"))
                for entry in existing
            }

    return set()  # Return empty set if result file does not exist


# Function to create an evaluation dictionary
def _create_evaluation_dictionary(model_number, run, config_name, duration, config, metrics, test_loss, test_accuracy):
    """
    Creates a full experiment record for evaluation and logging.

    Includes model identifiers, configuration parameters, training metrics, and test performance.

    Args:
        model_number (int): The model variant ID.
        run (int): The experiment run ID.
        config_name (str): The name of the configuration file used.
        duration (float): Total training and evaluation duration in seconds.
        config (Config): Loaded configuration object.
        metrics (dict): Training and validation metrics extracted after training.
        test_loss (float): Final test set loss.
        test_accuracy (float): Final test set accuracy.

    Returns:
        dict: Complete evaluation result to log or store.
    """

    # Print header for function execution
    print("\n🎯  _create_evaluation_dictionary")

    return {
        "model": model_number,
        "run": run,
        "config_name": config_name,
        "date": datetime.datetime.now().strftime("%Y-%m-%d"),
        "time": datetime.datetime.now().strftime("%H:%M:%S"),
        "duration": str(datetime.timedelta(seconds=int(duration))),
        "parameters": {
            "LIGHT_MODE": config.LIGHT_MODE,
            "AUGMENT_MODE": {
                "enabled": config.AUGMENT_MODE["enabled"],
                "random_crop": config.AUGMENT_MODE.get("random_crop", False),
                "random_flip": config.AUGMENT_MODE.get("random_flip", False),
                "cutout": config.AUGMENT_MODE.get("cutout", False)
            },
            "L2_MODE": {
                "enabled": config.L2_MODE["enabled"],
                "lambda": config.L2_MODE["lambda"]
            },
            "DROPOUT_MODE": {
                "enabled": config.DROPOUT_MODE["enabled"],
                "rate": config.DROPOUT_MODE["rate"]
            },
            "OPTIMIZER": {
                "type": config.OPTIMIZER["type"],
                "learning_rate": config.OPTIMIZER["learning_rate"],
                "momentum": config.OPTIMIZER.get("momentum", 0.0)
            },
            "SCHEDULE_MODE": {
                "enabled": config.SCHEDULE_MODE["enabled"],
                "warmup_epochs": config.SCHEDULE_MODE.get("warmup_epochs", 0),
                "factor": config.SCHEDULE_MODE.get("factor", None),
                "patience": config.SCHEDULE_MODE.get("patience", None),
                "min_lr": config.SCHEDULE_MODE.get("min_lr", None)
            },
            "EARLY_STOP_MODE": {
                "enabled": config.EARLY_STOP_MODE["enabled"],
                "patience": config.EARLY_STOP_MODE.get("patience", None),
                "restore_best_weights": config.EARLY_STOP_MODE.get("restore_best_weights", None)
            },
            "AVERAGE_MODE": {
                "enabled": config.AVERAGE_MODE["enabled"],
                "start_epoch": config.AVERAGE_MODE.get("start_epoch", None)
            },
            "TTA_MODE": {
                "enabled": config.TTA_MODE["enabled"],
                "runs": config.TTA_MODE.get("runs", 1)
            },
            "MIXUP_MODE": {
                "enabled": config.MIXUP_MODE["enabled"],
                "alpha": config.MIXUP_MODE["alpha"]
            },

            "EPOCHS_COUNT": config.EPOCHS_COUNT,
            "BATCH_SIZE": config.BATCH_SIZE
        },
        "min_train_loss": metrics["min_train_loss"],
        "min_train_loss_epoch": metrics["min_train_loss_epoch"],
        "max_train_acc": metrics["max_train_acc"],
        "max_train_acc_epoch": metrics["max_train_acc_epoch"],
        "min_val_loss": metrics.get("min_val_loss"),
        "min_val_loss_epoch": metrics.get("min_val_loss_epoch"),
        "max_val_acc": metrics.get("max_val_acc"),
        "max_val_acc_epoch": metrics.get("max_val_acc_epoch"),
        "final_test_loss": test_loss,
        "final_test_acc": test_accuracy
    }



# Function to recover training history
def _recover_training_history(config, run_id):
    """
    Attempts to load training history from disk using a unique run identifier.

    This serves as a fallback when training is resumed but the in-memory history
    object is unavailable. The run_id string (e.g. "m9_r1_default") is used to locate
    the corresponding checkpoint directory and retrieve the saved history.

    Args:
        config (Config): Configuration object with directory paths.
        run_id (str): Unique identifier for the run, formatted as 'm<model>_r<run>_<config>'.

    Returns:
        object | dict: Dummy object with `.history` if successful; empty dict otherwise.
    """

    # Print header for function execution
    print("\n🎯  _recover_training_history")

    history_file = config.CHECKPOINT_PATH / run_id / "history.json"

    if history_file.exists():
        try:
            with open(history_file, "r") as f:
                history_data = json.load(f)
                class DummyHistory: pass
                h = DummyHistory()
                h.history = history_data
                return h  # Return dummy object with recovered history
        except Exception as e:
            print(f"\n⚠️  Failing to recover training history:\n{e}")
    else:
        print(f"\n⚠️  Failing to find history for {run_id}")

    return {}  # Return empty dict as fallback if recovery fails


# Function to initialize logging
def _initialize_logging(timestamp):
    """
    Sets up dual logging to both terminal and a timestamped log file.

    Redirects stdout and stderr using a Tee class for simultaneous output.
    Also prepares the result file path for storing structured outputs.

    Args:
        timestamp (str): Unique timestamp used to name log and result files.

    Returns:
        tuple:
            - log_stream (IO): File stream object for dual logging.
            - result_file (Path): Path to structured result output file.
            - all_results_list (list): In-memory result buffer.

    """


    # Print header for function execution
    print("\n🎯  _initialize_logging")

    # Ensure log directory exists
    CONFIG.LOG_PATH.mkdir(parents=True, exist_ok=True)

    # Create new log file with timestamp
    log_file = CONFIG.LOG_PATH / f"log_{timestamp}.txt"
    log_stream = open(log_file, "a", buffering=1)

    # Redirect output streams to both terminal and log file
    sys.stdout = Tee(sys.__stdout__, log_stream)
    sys.stderr = Tee(sys.__stderr__, log_stream)

    # Confirm logging path
    print(f"\n📜  Logging experiment output:\n{log_file}", flush=True)

    # Ensure result directory exists and define result file path
    CONFIG.RESULT_PATH.mkdir(parents=True, exist_ok=True)
    result_file = CONFIG.RESULT_PATH / f"result_{timestamp}.json"

    return log_stream, result_file, []  # Return handles for logging and results


# Function to ensure output directories
def _ensure_output_directories(config):
    """
    Ensures existence of all required experiment output directories.

    Automatically creates the following paths if missing:
    - LOG_PATH
    - CHECKPOINT_PATH
    - RESULT_PATH
    - MODEL_PATH
    - ERROR_PATH

    Args:
        config (Config): The experiment configuration object with directory paths.

    Returns:
        None
    """

    # Print header for function execution
    print("\n🎯  _ensure_output_directories")

    print(f"\n📂  Ensuring output directories")  # Confirm creation or existence

    # Iterate through each required directory path and ensure it exists
    for path in [
        config.LOG_PATH,
        config.CHECKPOINT_PATH,
        config.RESULT_PATH,
        config.MODEL_PATH,
        config.ERROR_PATH
    ]:
        path.mkdir(parents=True, exist_ok=True)  # Create directory if missing
        print(f"{path}")  # Confirm creation or existence


# # Class for parallel writing in stdout and log
class Tee:
    """
    Custom class to duplicate stdout/stderr to multiple destinations.

    Used to stream logs both to the terminal and a file concurrently.
    """

    def __init__(self, *streams):
        """
        Initialize with multiple stream objects (e.g., sys.stdout and a file handle).

        Args:
            *streams: Arbitrary number of writable stream objects.
        """

        self.streams = streams

    def write(self, data):
        """
        Write data to all attached streams.

        Args:
            data (str): The string data to write.
        """

        for s in self.streams:
            try:
                s.write(data)      # Attempt to write to the stream
                s.flush()         # Ensure the data is pushed immediately
            except Exception as e:
                print(f"\n❌ Tee write failed: {e}")  # Handle write failure gracefully

    def flush(self):
        """
        Flush all streams to ensure complete write.
        """

        for s in self.streams:
            s.flush()


# Print module confirmation
print("\n✅  experiment.py successfully executed")
