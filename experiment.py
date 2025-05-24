# Import standard libraries
import datetime
import sys
import json
import time
import traceback

# Import project-specific libraries
from config import CONFIG
from data import load_dataset
from evaluate import extract_history_metrics
from log import log_to_json
from model import build_model
from train import train_model


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
    """

    # Print header for function execution
    print("\n🎯  run_pipeline")

    # Generate timestamp for consistent filenames
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Load config from the first entry to bootstrap directory setup
    first_model, first_config_name = pipeline[0]
    first_config_path = CONFIG.CONFIG_PATH / f"{first_config_name}.json"
    first_config = CONFIG.load_config(first_config_path)

    # Initialize logging and result tracking
    log_file, log_stream, result_file, all_results = _initialize_logging(timestamp)

    try:
        # Load previously completed (model, config) combinations
        completed_triplets = _load_previous_results(result_file, all_results)

        # Dictionary to track how many times each model_number has been run
        model_run_counter = {}

        # Loop through each (model_number, config_name) in the pipeline
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

        # Save all accumulated results after pipeline execution
        with open(result_file, "w") as jf:
            json.dump(all_results, jf, indent=2)

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

    # Create output folders if missing
    _ensure_output_directories(config)

    # Skip if already completed
    if (model_number, run, config_name) in completed_triplets:
        print(f"\n⏩  Skipping experiment m{model_number}_r{run} with '{config_name}'")
        return result_file  # Return early since result already exists

    # Announce launch
    print(f"\n🚀  Launching experiment m{model_number}_r{run} with '{config_name}'")
    start_time = time.time()

    try:
        # Load dataset for this model variant
        train_data, train_labels, test_data, test_labels = load_dataset(model_number, config)

        # Build model architecture
        model = build_model(model_number, config)

        # Train model (resumable)
        trained_model, history, resumed = train_model(
            train_data, train_labels,
            model, model_number, run, config_name,
            timestamp, config
        )

        # Recover history if training was resumed and history is missing
        if resumed and (history is None or not hasattr(history, "history")):
            history = _recover_training_history(config, model_number, run, config_name)

        # Extract training/validation metrics
        metrics = extract_history_metrics(history)

        # Evaluate final model on test data
        final_test_loss, final_test_acc = trained_model.evaluate(
            test_data, test_labels,
            batch_size=config.BATCH_SIZE, verbose=0
        )

        # Build evaluation dictionary for logging
        evaluation = _create_evaluation_dictionary(
            model_number, run, config_name,
            time.time() - start_time, config,
            metrics, final_test_loss, final_test_acc
        )

        # Print summary to console
        print("\n📊  Dumping experiment results:")
        print(json.dumps([evaluation], indent=2))

        # Append to in-memory result list and save to disk
        all_results.append(evaluation)
        with open(result_file, "w") as jf:
            json.dump(all_results, jf, indent=2)

        print(f"\n✅   m{model_number} run {run} with '{config_name}' successfully executed")

    except Exception as e:
        # On failure, log error details to error file
        log_to_json(
            config.ERROR_PATH, key=None,
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
    Function to load existing results from disk and extend in-memory results list.

    Args:
        result_file (Path): Path to the result.json file
        all_results (list): Reference to the current in-memory results list

    Returns:
        set: A set of (model_number, run, config_name) triplets for deduplication
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
    Function to build structured evaluation dictionary.

    Combines model metadata, training config, runtime metrics, and evaluation results
    into a single dictionary for logging.

    Args:
        model_number (int): Model variant used
        run (int): Run ID for the model
        config_name (str): Config file name
        duration (float): Time taken for the run (seconds)
        config (Config): Loaded config object
        metrics (dict): Extracted training/validation metrics
        test_loss (float): Final test loss
        test_accuracy (float): Final test accuracy

    Returns:
        dict: Structured evaluation result
    """

    # Print header for function execution
    print("\n🎯  _create_evaluation_dictionary")

    return {
        "model": model_number,
        "run": run,
        "config": config_name,
        "date": datetime.datetime.now().strftime("%Y-%m-%d"),
        "time": datetime.datetime.now().strftime("%H:%M:%S"),
        "duration": str(datetime.timedelta(seconds=int(duration))),
        "parameters": {
            "LIGHT_MODE": config.LIGHT_MODE,
            "AUGMENT_MODE": config.AUGMENT_MODE,

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

            "SCHEDULE_MODE": config.SCHEDULE_MODE["enabled"],
            "EARLY_STOP_MODE": config.EARLY_STOP_MODE["enabled"],

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
def _recover_training_history(config, model_number, run, config_name):
    """
    Attempts to recover the training history from disk as a fallback.

    If the training was resumed and no history object is in memory, this function
    reads the stored history.json file and wraps it in a dummy object.

    Args:
        config (Config): Configuration object with checkpoint path
        model_number (int): Model identifier
        run (int): Run number
        config_name (str): Config name used during training

    Returns:
        object or dict: An object with .history attribute or empty dict if not found
    """

    # Print header for function execution
    print("\n🎯  _recover_training_history")

    history_file = config.CHECKPOINT_PATH / f"m{model_number}_r{run}_{config_name}/history.json"

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
        print(f"\n⚠️  Failing to find history for m{model_number}_r{run}_{config_name}")

    return {}  # Return empty dict as fallback if recovery fails


# Function to initialize logging
def _initialize_logging(timestamp):
    """
    Initializes dual logging to both console and log file.

    Creates a timestamped log file in LOG_PATH and redirects stdout and stderr
    using the Tee class. Also prepares the result file path and structure.

    Args:
        timestamp (str): Current timestamp used for naming files.

    Returns:
        tuple: (log_file, log_stream, result_file, all_results_list)
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

    return log_file, log_stream, result_file, []  # Return handles for logging and results


# Function to ensure output directories
def _ensure_output_directories(config):
    """
    Ensures all output directories required for the experiment exist.

    Creates directories if they do not exist for:
    - LOG_PATH
    - CHECKPOINT_PATH
    - RESULT_PATH
    - MODEL_PATH
    - ERROR_PATH

    Args:
        config (Config): Configuration object with path definitions
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
    Custom class to duplicate output streams.

    Writes any data sent to it to all provided output streams,
    typically the terminal and a log file simultaneously.
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
