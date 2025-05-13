# Import standard libraries
import shutil
import datetime
import sys
import json

# Import project-specific libraries
from config import CONFIG
from data import load_dataset
from evaluate import evaluate_model
from model import build_model
from train import train_model


# Function to run experiments on models
def run_experiment(model_numbers=0, runs=1):
    """
    Runs one or more training experiments, logs output to both terminal and log file,
    and saves results to a timestamped JSON file.

    Args:
        model_numbers (int or tuple): Single model ID or a range of IDs (e.g. 1 or (1, 3)).
        runs (int): Number of repetitions per model (default: 1).
    """

    # Print header for function execution
    print("\n🎯 run_experiment")

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
        all_results = []

        # Normalize model_numbers input
        if isinstance(model_numbers, int):
            model_numbers = [model_numbers]
        elif isinstance(model_numbers, tuple) and len(model_numbers) == 2:
            model_numbers = list(range(model_numbers[0], model_numbers[1] + 1))

        # Iterate through all specified models and runs
        for model_number in model_numbers:
            for run in range(1, runs + 1):
                print(f"\n🚀 Launching m{model_number} ({run}/{runs}) ...")

                # Load dataset
                train_data, train_labels, test_data, test_labels = load_dataset(model_number)

                # Build and train model
                model = build_model(model_number)
                trained_model, history = train_model(
                    train_data, train_labels, model, model_number, timestamp
                )

                # Record results for current run
                evaluation = evaluate_model(trained_model, history, test_data, test_labels)

                evaluation.update({
                    "model": model_number,
                    "time": datetime.datetime.now().strftime("%H:%M:%S"),
                    "layers": len(trained_model.layers),
                    "optimizer": type(trained_model.optimizer).__name__
                })

                # Remove non-serializable fields
                evaluation_clean = {k: v for k, v in evaluation.items() if k != "predictions"}

                # Print clean JSON to terminal
                print("\n📊 Summary JSON:")
                print(json.dumps([evaluation_clean], indent=2))

                # Save clean JSON to results
                all_results.append(evaluation_clean)


        # Save all run results into a timestamped result file
        with open(result_file, "w") as jf:
            json.dump(all_results, jf, indent=2)

    # Restore standard output and close log stream
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

    # Define the folders to target along with their associated CLEAN_* flags
    targets = [
        (CONFIG.LOG_PATH, CONFIG.CLEAN_LOG),
        (CONFIG.CHECKPOINT_PATH, CONFIG.CLEAN_CHECKPOINT),
        (CONFIG.RESULT_PATH, CONFIG.CLEAN_RESULT),
        (CONFIG.MODEL_PATH, CONFIG.CLEAN_MODEL),
    ]

    # Iterate over each folder and clean it if the flag is enabled
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
