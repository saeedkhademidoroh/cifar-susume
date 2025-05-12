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


# Function to run training experiments and optionally log and save results
def run_experiment(model_numbers=0, runs=1):
    """
    Runs one or more training experiments, logs terminal output to file
    (if LIGHT_MODE is disabled), and saves results as a timestamped JSON file.

    Args:
        model_numbers (int or tuple): Model ID(s) to run (e.g. 1 or (1, 3)).
        runs (int): Number of repetitions per model (default is 1).
    """
    print("\n🎯 run_experiment")

    # Clean unnecessary data
    clean_old_output()

    # Generate timestamp for result/log filenames
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Redirect output to log file
    if not CONFIG.LIGHT_MODE:
        CONFIG.LOG_PATH.mkdir(parents=True, exist_ok=True)
        log_file = CONFIG.LOG_PATH / f"log_{timestamp}.txt"
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        sys.stdout = open(log_file, "a", buffering=1)  # line-buffered
        sys.stderr = sys.stdout
        print(f"\n📝 Logging:\n{log_file}\n", flush=True)
    else:
        log_file = None

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
                    train_data, train_labels, model, model_number
                )

                # Record results for current run
                eval_result = evaluate_model(trained_model, history, test_data, test_labels)
                result = {
                    "model": model_number,
                    "time": datetime.datetime.now().strftime("%H:%M:%S"),
                    "layers": len(trained_model.layers),
                    "optimizer": type(trained_model.optimizer).__name__,
                    "val_accuracy": eval_result.get("max_val_acc"),
                    "train_accuracy": eval_result.get("max_train_acc"),
                    "val_loss": eval_result.get("min_val_loss"),
                    "train_loss": eval_result.get("min_train_loss"),
                    "test_accuracy": eval_result.get("final_test_accuracy"),
                    "test_loss": eval_result.get("final_test_loss")
                }

                all_results.append(result)

        # Save all run results into a timestamped result file
        with open(result_file, "w") as jf:
            json.dump(all_results, jf, indent=2)

    finally:
        # Restore standard output if redirected
        if log_file:
            sys.stdout.close()
            sys.stdout = original_stdout
            sys.stderr = original_stderr


# Function to clean output
def clean_old_output():
    """
    Cleans output directories specified in the configuration.

    Each directory is cleaned only if its corresponding CLEAN_* flag is set to True.
    """
    print("\n🎯 clean_old_output")

    # Define the folders to target along with their associated CLEAN_* flags
    targets = [
        (CONFIG.LOG_PATH, CONFIG.CLEAN_LOG),
        (CONFIG.CHECKPOINT_PATH, CONFIG.CLEAN_CHECKPOINT),
        (CONFIG.RESULT_PATH, CONFIG.CLEAN_RESULT),
    ]

    # Iterate over each folder and clean it if the flag is enabled
    for path, clean_flag in targets:
        if clean_flag and path.exists():
            try:
                shutil.rmtree(path)
                print(f"\n🗑️  Cleaned:\n{path}")
            except Exception as e:
                print(f"\n❌ Exception:\n{path}\n{e}\n")



# Print confirmation message
print("\n✅ experiment.py successfully executed")
