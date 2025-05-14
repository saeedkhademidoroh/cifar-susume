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
from config import CONFIG, Config
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

    # Infer number of runs if not provided, based on config_map
    if runs is None and config_map:
        runs = max(
            max(runs_dict.keys()) for runs_dict in config_map.values()
            if isinstance(runs_dict, dict) and runs_dict
        )
    elif runs is None:
        runs = 1

    # Fallback: apply default config to all runs if no config_map is given
    if config_map is None:
        config_map = {
            model: {1: CONFIG.CONFIG_PATH / "default.json"}
            for model in model_numbers
        }

    # Clean old output
    clean_old_output()

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

        # Track already completed (model, run) pairs
        completed_pairs = {(entry["model"], entry.get("run", 1)) for entry in all_results}

        # Normalize model_numbers input
        if isinstance(model_numbers, int):
            model_numbers = [model_numbers]
        elif isinstance(model_numbers, tuple) and len(model_numbers) == 2:
            model_numbers = list(range(model_numbers[0], model_numbers[1] + 1))

        # Iterate through all specified models and runs
        for model_number in model_numbers:
            for run in range(1, runs + 1):
                if (model_number, run) in completed_pairs:
                    print(f"\n⏩ Skipping m{model_number}_r{run} — already completed")
                    continue

                print(f"\n🚀 Launching m{model_number} ({run}/{runs}) ...")

                # Load custom config if specified for this model and run
                run_config_path = config_map.get(model_number, {}).get(run) if config_map else None

                if run_config_path:
                    custom_config_path = Path(run_config_path)
                    dynamic_config = Config.load_from_custom_path(custom_config_path)
                    config_name = custom_config_path.stem
                else:
                    dynamic_config = CONFIG
                    config_name = "default"

                # Track training start time
                start_time = time.time()

                try:
                    # Load dataset
                    train_data, train_labels, test_data, test_labels = load_dataset(model_number)

                    # Build and train model
                    model = build_model(model_number)
                    trained_model, history, resumed = train_model(
                        train_data, train_labels, model, model_number, run, timestamp
                    )


                    if resumed and history is None:
                        print(f"\n⚠️ Resumed m{model_number} — no training history available")
                        metrics = {
                            "min_train_loss": None,
                            "min_train_loss_epoch": None,
                            "max_train_acc": None,
                            "max_train_acc_epoch": None,
                            "min_val_loss": None,
                            "min_val_loss_epoch": None,
                            "max_val_acc": None,
                            "max_val_acc_epoch": None
                        }
                        final_test_loss, final_test_accuracy = trained_model.evaluate(
                            test_data, test_labels, batch_size=dynamic_config.BATCH_SIZE, verbose=0)
                    else:
                        metrics = extract_history_metrics(history)
                        final_test_loss, final_test_accuracy = trained_model.evaluate(
                            test_data, test_labels, batch_size=dynamic_config.BATCH_SIZE, verbose=0)

                    # Construct structured evaluation dictionary
                    evaluation = {
                        "model": model_number,
                        "config": config_name,
                        "time": datetime.datetime.now().strftime("%H:%M:%S"),
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
                    # Log the error to a timestamped file in ERROR_PATH
                    log_to_json(
                        CONFIG.ERROR_PATH,           # Write into artifact/error
                        key=None,                    # Not needed for error log
                        record={
                            "model": model_number,   # Include model identifier
                            "run": run,              # Include run index
                            "config_name": config_name,  # Identify config source
                            "error": str(e),         # Main error message
                            "exception_type": type(e).__name__,  # Error type
                            "trace": traceback.format_exc()      # Full traceback
                        },
                        error=True                   # Trigger error log mode
                    )

                    # Re-raise the error after logging for visibility
                    raise



        # Final write
        with open(result_file, "w") as jf:
            json.dump(all_results, jf, indent=2)

    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        log_stream.close()


# Function to clean output
def clean_old_output():
    """
    Cleans output directories specified in the configuration.

    Each directory is cleaned only if its corresponding CLEAN_* flag is set to True.
    """

    # Print header for function execution
    print("\n🎯 clean_old_output")

    targets = [
        (CONFIG.LOG_PATH, CONFIG.CLEAN_LOG),
        (CONFIG.CHECKPOINT_PATH, CONFIG.CLEAN_CHECKPOINT),
        (CONFIG.RESULT_PATH, CONFIG.CLEAN_RESULT),
        (CONFIG.MODEL_PATH, CONFIG.CLEAN_MODEL),
        (CONFIG.HISTORY_PATH, CONFIG.CLEAN_HISTORY)
    ]

    for path, clean_flag in targets:
        if clean_flag and path.exists():
            try:
                shutil.rmtree(path)
                print(f"\n🗑️  Cleaning:\n{path}")
            except Exception as e:
                print(f"\n❌ Exception:\n{path}\n{e}\n")


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
