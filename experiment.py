# Import standard libraries
import shutil
import datetime
import sys
import json
import time
from collections import Counter
from pathlib import Path
import traceback

# Import project-specific libraries
from config import CONFIG
from data import load_dataset
from evaluate import extract_history_metrics
from model import build_model
from train import train_model
from log import log_to_json


# Function to run experiments on models
def run_experiment(model_numbers=0, runs=None, config_map=None):
    """
    Runs one or more training experiments, logs output to both terminal and log file,
    and saves results to a timestamped JSON file.

    Args:
        model_numbers (int or tuple): Single model ID or a range of IDs (e.g. 1 or (1, 3)).
        runs (int): Number of repetitions per model (default: 1).
        config_map (dict): Optional mapping of model_number to config file path.
    """

    # Print header for function execution
    print("\n🎯 run_experiment")

    # Normalize model_numbers to a list of integers
    if isinstance(model_numbers, int):
        model_numbers = [model_numbers]
    elif isinstance(model_numbers, tuple) and len(model_numbers) == 2:
        model_numbers = list(range(model_numbers[0], model_numbers[1] + 1))
    elif not isinstance(model_numbers, list):
        raise ValueError(f"❌ Invalid model_numbers: {model_numbers}")

    # If config_map exists, filter it down to relevant models
    if config_map:
        config_map = {m: config_map[m] for m in model_numbers if m in config_map}
        if not config_map:
            raise ValueError(f"❌ No matching models found in config_map for: {model_numbers}")
    else:
        # Fallback to default config for all models
        config_map = {
            model: {1: CONFIG.CONFIG_PATH / "default.json"}
            for model in model_numbers
        }

    # Infer number of runs if not explicitly provided
    if runs is None:
        runs = max(
            max(runs_dict.keys(), default=0)
            for runs_dict in config_map.values()
            if isinstance(runs_dict, dict)
        )

    # Generate timestamp for result/log filenames
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Setup dual logging: terminal + log file
    CONFIG.LOG_PATH.mkdir(parents=True, exist_ok=True)
    log_file = CONFIG.LOG_PATH / f"log_{timestamp}.txt"
    log_stream = open(log_file, "a", buffering=1)
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = Tee(sys.stdout, log_stream)
    sys.stderr = Tee(sys.stderr, log_stream)
    print(f"\n📝 Logging:\n{log_file}", flush=True)

    try:
        # Create result path and file
        CONFIG.RESULT_PATH.mkdir(parents=True, exist_ok=True)
        result_file = CONFIG.RESULT_PATH / f"result_{timestamp}.json"

        # Load existing results if file exists (e.g., on resume)
        if result_file.exists():
            with open(result_file, "r") as jf:
                all_results = json.load(jf)
        else:
            all_results = []

        # Track already completed (model, run, config) based on logged results
        completed_triplets = {
            (entry["model"], entry.get("run", 1), entry.get("config", "default"))
            for entry in all_results
        }

        cleaned_configs = set()

        # Iterate through all specified models and their runs (with auto-incrementing run numbers)
        for model_number in model_numbers:
            model_runs = config_map.get(model_number, {})
            for run_counter in sorted(model_runs.keys()):

                # Resolve config name before checking skip list
                run_config_path = config_map.get(model_number, {}).get(run_counter) if config_map else None
                if run_config_path:
                    custom_config_path = Path(run_config_path)
                    config_name = custom_config_path.stem
                    dynamic_config = CONFIG.load_custom_config(custom_config_path)
                else:
                    config_name = "default"
                    dynamic_config = CONFIG

                clean_old_output(dynamic_config)
                ensure_output_paths(dynamic_config)

                # Skip if this model-run-config combination already exists in results
                if (model_number, run_counter, config_name) in completed_triplets:
                    print(f"\n⏩ Skipping m{model_number}_r{run_counter} — already logged with config '{config_name}'")
                    continue

                # Determine total declared runs for this model from config_map
                total_runs = len(config_map.get(model_number, {}))

                # Skip if this model-run-config combination already exists in results
                if (model_number, run_counter, config_name) in completed_triplets:
                    print(f"\n⏩ Skipping m{model_number}_r{run_counter} — already logged with config '{config_name}'")
                    continue

                # Announce this run is launching
                print(f"\n🚀 Launching m{model_number}_r{run_counter} ({run_counter}/{total_runs}) ...")
                run = run_counter

                # Track training start time (currently unused, may be used for logging or profiling later)
                start_time = time.time()
                duration = None

                try:
                    # Load dataset
                    train_data, train_labels, test_data, test_labels = load_dataset(model_number)

                    # Build and train model
                    model = build_model(model_number)
                    trained_model, history, resumed = train_model(
                        train_data, train_labels, model, model_number, run, dynamic_config, timestamp
                    )

                    # Handle resumed runs with no in-memory history
                    if resumed and (history is None or not hasattr(history, "history") or "loss" not in history.history):
                        print(f"\n⚠️  Resumed m{model_number} — attempting to reload history")
                        history_file = CONFIG.CHECKPOINT_PATH / f"m{model_number}_r{run}_{config_name}/history.json"
                        if history_file.exists():
                            try:
                                with open(history_file, "r") as f:
                                    history_data = json.load(f)
                                    class DummyHistory: pass
                                    history = DummyHistory()
                                    history.history = history_data
                                print(f"\n📄 Loaded history from checkpoint for m{model_number}_r{run}_{config_name}")
                                metrics = extract_history_metrics(history)
                            except Exception as e:
                                print(f"\n⚠️  Failed to load or parse history:\n{e}")
                                metrics = {
                                    "min_train_loss": None, "min_train_loss_epoch": None,
                                    "max_train_acc": None, "max_train_acc_epoch": None,
                                    "min_val_loss": None, "min_val_loss_epoch": None,
                                    "max_val_acc": None, "max_val_acc_epoch": None
                                }
                        else:
                            print(f"\n⚠️  No history file found for m{model_number}_r{run}_{config_name}")
                            metrics = {
                                "min_train_loss": None, "min_train_loss_epoch": None,
                                "max_train_acc": None, "max_train_acc_epoch": None,
                                "min_val_loss": None, "min_val_loss_epoch": None,
                                "max_val_acc": None, "max_val_acc_epoch": None
                            }

                        final_test_loss, final_test_accuracy = trained_model.evaluate(
                            test_data, test_labels, batch_size=dynamic_config.BATCH_SIZE, verbose=0)

                    else:
                        # Standard case — extract metrics from live history
                        metrics = extract_history_metrics(history)
                        final_test_loss, final_test_accuracy = trained_model.evaluate(
                            test_data, test_labels, batch_size=dynamic_config.BATCH_SIZE, verbose=0)

                    # Done with training — measure time now
                    duration = time.time() - start_time
                    print(f"\n⏱️  Duration: {str(datetime.timedelta(seconds=int(duration)))}")

                    # Construct structured evaluation dictionary
                    evaluation = {
                        "model": model_number,
                        "run": run,
                        "config": config_name,
                        "date": datetime.datetime.now().strftime("%Y-%m-%d"),
                        "time": datetime.datetime.now().strftime("%H:%M:%S"),
                        "duration": str(datetime.timedelta(seconds=int(duration))) if duration is not None else None,
                        "parameters": {
                            "EPOCHS_COUNT": dynamic_config.EPOCHS_COUNT,
                            "BATCH_SIZE": dynamic_config.BATCH_SIZE
                        },
                        "min_train_loss": metrics["min_train_loss"],
                        "min_train_loss_epoch": metrics["min_train_loss_epoch"],
                        "max_train_acc": metrics["max_train_acc"],
                        "max_train_acc_epoch": metrics["max_train_acc_epoch"],
                        "min_val_loss": metrics.get("min_val_loss"),
                        "min_val_loss_epoch": metrics.get("min_val_loss_epoch"),
                        "max_val_acc": metrics.get("max_val_acc"),
                        "max_val_acc_epoch": metrics.get("max_val_acc_epoch"),
                        "final_test_loss": final_test_loss,
                        "final_test_accuracy": final_test_accuracy
                    }

                    # Print JSON-formatted summary
                    print("\n📊 Summary JSON:")
                    print(json.dumps([evaluation], indent=2))

                    # Save result
                    all_results.append(evaluation)
                    with open(result_file, "w") as jf:
                        json.dump(all_results, jf, indent=2)

                    print(f"\n✅ m{model_number} run {run} completed and result logged")

                # Catch and log any unexpected errors per run
                except Exception as e:
                    log_to_json(
                        CONFIG.ERROR_PATH,
                        key=None,
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
                    raise

        # Final write to result JSON file
        with open(result_file, "w") as jf:
            json.dump(all_results, jf, indent=2)

    finally:
        # Restore stdout/stderr and close log stream
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        log_stream.close()


# Function to ensure all required output directories exist
def ensure_output_paths(config):
    """
    Ensures that all essential output directories defined in the configuration
    are recreated after cleanup. Logs each path ensured.
    """

    # Print header for function execution
    print("\n🎯 ensure_output_paths")

    paths = [
        config.LOG_PATH,
        config.CHECKPOINT_PATH,
        config.RESULT_PATH,
        config.MODEL_PATH,
        config.ERROR_PATH
    ]

    for path in paths:
        path.mkdir(parents=True, exist_ok=True)
        print(f"\n📂 Ensured:\n{path}")


# Function to ensure all required output directories exist
def ensure_output_paths(config):
    """
    Ensures that all essential output directories defined in the configuration
    are recreated after cleanup. Logs each path ensured.
    """

    # Print header for function execution
    print("\n🎯 ensure_output_paths")

    paths = [
        config.LOG_PATH,
        config.CHECKPOINT_PATH,
        config.RESULT_PATH,
        config.MODEL_PATH,
        config.ERROR_PATH
    ]

    for path in paths:
        path.mkdir(parents=True, exist_ok=True)
        print(f"\n📂 Ensured:\n{path}")


# Function to clean output
def clean_old_output(config):
    """
    Cleans output directories specified in the configuration.

    Each directory is cleaned only if its corresponding CLEAN_* flag is set to True.
    """

    # Print header for function execution
    print("\n🎯 clean_old_output")

    # Define cleanup targets and corresponding flags
    targets = [
        (config.LOG_PATH, config.CLEAN_LOG),
        (config.CHECKPOINT_PATH, config.CLEAN_CHECKPOINT),
        (config.RESULT_PATH, config.CLEAN_RESULT),
        (config.MODEL_PATH, config.CLEAN_MODEL),
        (config.ERROR_PATH, config.CLEAN_ERROR),
    ]

    # Iterate through paths and clean if flag is set and path exists
    for path, clean_flag in targets:
        if clean_flag and path.exists():
            try:
                shutil.rmtree(path)
                print(f"\n🗑️  Cleaning:\n{path}")
            except Exception as e:
                print(f"\n❌ Exception:\n{path}\n{e}")
        else:
            print(f"\n⚪ Skipped: (clean_flag={clean_flag}, exists={path.exists()})\n{path}")


# Class to tee output to both terminal and log file
class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()


# Print confirmation message
print("\n✅ experiment.py successfully executed")
