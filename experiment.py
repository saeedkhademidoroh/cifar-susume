# Import standard libraries
import shutil
import datetime
import sys
import json
import time
import traceback
from pathlib import Path

# Import project-specific libraries
from config import CONFIG
from data import dispatch_load_dataset
from evaluate import extract_history_metrics
from model import build_model
from train import train_model
from log import log_to_json


# Function to run all experiments given a pipeline of (model, config_name) pairs
def run_pipeline(pipeline):
    """
    Main function to execute a list of model/config experiments.

    Args:
        pipeline (list of tuples): Each entry is (model_number: int, config_name: str)
    """
    print("\n🎯 run_pipeline")

    # Generate timestamp for naming logs/results
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Pre-cleaning based on the first config in the pipeline
    first_model, first_config_name = pipeline[0]
    first_config_path = CONFIG.CONFIG_PATH / f"{first_config_name}.json"
    first_config = CONFIG.load_config(first_config_path)

    # Create output folders if missing
    ensure_output_paths(first_config)

    # Start logging after path check
    log_file, log_stream, result_file, all_results = _initialize_logging(timestamp)
    print(f"\n📝 Log file ready at: {log_file}")

    try:
        # Load completed entries from result file to skip them
        completed_triplets = _load_previous_results(result_file, all_results)

        # ✅ Initialize per-model run tracker
        model_run_counter = {}

        # Iterate through each pipeline task
        for i, (model_number, config_name) in enumerate(pipeline):
            print(f"\n⚙️ Pipeline Entry {i+1}/{len(pipeline)} ---")
            config_path = CONFIG.CONFIG_PATH / f"{config_name}.json"

            # ✅ Track run per model (not per pipeline index)
            if model_number not in model_run_counter:
                model_run_counter[model_number] = 1
            else:
                model_run_counter[model_number] += 1

            run = model_run_counter[model_number]

            # Run single experiment
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

        # Final write of results
        with open(result_file, "w") as jf:
            json.dump(all_results, jf, indent=2)

    finally:
        # Restore stdout and close log file
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        if log_stream:
            log_stream.close()


# Executes a single training run given model, config, and run ID
def _run_single_pipeline_entry(model_number, config_path, config_name, run, timestamp, completed_triplets, all_results, result_file):
    # Load dynamic configuration for this run
    config = CONFIG.load_config(config_path)

    # Create output folders if missing
    ensure_output_paths(config)

    # Skip if already completed
    if (model_number, run, config_name) in completed_triplets:
        print(f"\n⏩ Skipping m{model_number}_r{run} — already logged with config '{config_name}'")
        return result_file

    # Announce launch
    print(f"\n🚀 Launching m{model_number}_r{run} with '{config_name}'")
    start_time = time.time()

    try:
        # Load dataset for this model
        train_data, train_labels, test_data, test_labels = dispatch_load_dataset(model_number, config)

        # Build model
        model = build_model(model_number)

        # Train model
        trained_model, history, resumed = train_model(
            train_data, train_labels,
            model, model_number, run, config_name,
            timestamp, config
        )

        # Recover history if training resumed without in-memory history
        if resumed and (history is None or not hasattr(history, "history")):
            history = _recover_history(config, model_number, run, config_name)

        # Extract metrics from training history
        metrics = extract_history_metrics(history)

        # Evaluate final model
        final_test_loss, final_test_accuracy = trained_model.evaluate(
            test_data, test_labels,
            batch_size=config.BATCH_SIZE, verbose=0
        )

        # Create result dict
        evaluation = _create_evaluation_dict(
            model_number, run, config_name,
            time.time() - start_time, config,
            metrics, final_test_loss, final_test_accuracy
        )

        # Print JSON result
        print("\n📊 Summary JSON:")
        print(json.dumps([evaluation], indent=2))

        # Save to results
        all_results.append(evaluation)
        with open(result_file, "w") as jf:
            json.dump(all_results, jf, indent=2)

        print(f"\n✅ m{model_number} run {run} completed and result logged")

    except Exception as e:
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
        raise

    return result_file



# Helper to load previous result records from file
def _load_previous_results(result_file, all_results):
    if result_file.exists():
        with open(result_file, "r") as jf:
            existing = json.load(jf)
            all_results.extend(existing)
            return {
                (entry["model"], entry.get("run", 1), entry.get("config", "default"))
                for entry in existing
            }
    return set()


# Construct result JSON from metrics and metadata
def _create_evaluation_dict(model_number, run, config_name, duration, config, metrics, loss, acc):
    return {
        "model": model_number,
        "run": run,
        "config": config_name,
        "date": datetime.datetime.now().strftime("%Y-%m-%d"),
        "time": datetime.datetime.now().strftime("%H:%M:%S"),
        "duration": str(datetime.timedelta(seconds=int(duration))),
        "parameters": {
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
        "final_test_loss": loss,
        "final_test_accuracy": acc
    }


# Recover training history from disk if needed
def _recover_history(config, model_number, run, config_name):
    history_file = config.CHECKPOINT_PATH / f"m{model_number}_r{run}_{config_name}/history.json"
    if history_file.exists():
        try:
            with open(history_file, "r") as f:
                history_data = json.load(f)
                class DummyHistory: pass
                h = DummyHistory()
                h.history = history_data
                return h
        except Exception as e:
            print(f"\n⚠️  Failed to load or parse history:\n{e}")
    else:
        print(f"\n⚠️  No history file found for m{model_number}_r{run}_{config_name}")
    return {}


# Setup logging and result path
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

    # Ensure log directory exists
    CONFIG.LOG_PATH.mkdir(parents=True, exist_ok=True)

    # Create new log file with timestamp
    log_file = CONFIG.LOG_PATH / f"log_{timestamp}.txt"
    log_stream = open(log_file, "a", buffering=1)

    # Redirect output streams to both terminal and log file
    sys.stdout = Tee(sys.__stdout__, log_stream)
    sys.stderr = Tee(sys.__stderr__, log_stream)

    # Confirm logging path
    print(f"\n📜 Logging:\n{log_file}", flush=True)

    # Ensure result directory exists and define result file path
    CONFIG.RESULT_PATH.mkdir(parents=True, exist_ok=True)
    result_file = CONFIG.RESULT_PATH / f"result_{timestamp}.json"

    # Return the logging and result handles
    return log_file, log_stream, result_file, []


# Ensures required output directories exist
def ensure_output_paths(config):
    print("\n🎯 ensure_output_paths")
    for path in [config.LOG_PATH, config.CHECKPOINT_PATH, config.RESULT_PATH, config.MODEL_PATH, config.ERROR_PATH]:
        path.mkdir(parents=True, exist_ok=True)
        print(f"\n📂 Ensured:\n{path}")


# Logging class to write to both stdout and log file
class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            try:
                s.write(data)
                s.flush()
            except Exception as e:
                print(f"\n❌ Tee write failed: {e}")

    def flush(self):
        for s in self.streams:
            s.flush()


# Print module confirmation
print("\n✅ experiment.py successfully executed")
