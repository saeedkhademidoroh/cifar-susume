# Import standard libraries
import shutil
import datetime
import sys
import json

# Import project-specific libraries
from config import CONFIG
from data import load_dataset
from model import build_model
from train import train_model


# Function to run training experiments and optionally log and save results
def run_experiment(model_numbers, runs=1):
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

    # Optional log file setup (only in full mode)
    if not CONFIG.LIGHT_MODE:
        CONFIG.LOG_PATH.mkdir(parents=True, exist_ok=True)
        log_file = CONFIG.LOG_PATH / f"log_{timestamp}.txt"
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        sys.stdout = open(log_file, "a")
        sys.stderr = sys.stdout
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
                train_data, train_labels, _, _ = load_dataset(model_number)

                # Build and train model
                model = build_model(model_number)
                trained_model, history = train_model(
                    train_data, train_labels, model, model_number
                )

                # Record results for current run
                result = collect_experiment_result(trained_model, history, model_number)
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


# Function to structure metrics and metadata from training
def collect_experiment_result(model, history, model_number):
    """
    Extracts relevant training results and model metadata.

    Args:
        model (Model): Trained Keras model.
        history (History): Training history object from model.fit().
        model_number (int): ID of the model variant.

    Returns:
        dict: Structured summary with accuracy, loss, and config details.
    """
    print("\n🎯 collect_experiment_result")

    return {
        "model": model_number,
        "time": datetime.datetime.now().strftime("%H:%M:%S"),
        "layers": len(model.layers),
        "optimizer": type(model.optimizer).__name__,
        "val_accuracy": round(max(history.history.get("val_accuracy", [0])), 4),
        "train_accuracy": round(max(history.history.get("accuracy", [0])), 4),
        "val_loss": round(min(history.history.get("val_loss", [0])), 4),
        "train_loss": round(min(history.history.get("loss", [0])), 4),
    }


# Function to remove output folders based on individual CLEAN_* flags
def clean_old_output():
    """
    Removes log, checkpoint, and result folders based on config flags:
        - CLEAN_LOG
        - CLEAN_CHECKPOINT
        - CLEAN_RESULT
    """
    print("\n🎯 clean_old_output")

    targets = [
        (CONFIG.LOG_PATH, CONFIG.CLEAN_LOG),
        (CONFIG.CHECKPOINT_PATH, CONFIG.CLEAN_CHECKPOINT),
        (CONFIG.RESULT_PATH, CONFIG.CLEAN_RESULT),
    ]

    for path, clean_flag in targets:
        if clean_flag and path.exists():
            try:
                shutil.rmtree(path)
                print(f"\n🗑️  Cleaned:\n{path}")
            except Exception as e:
                print(f"\n❌ Exception:\n{path}\n{e}\n")




# Confirmation message
print("\n✅ experiment.py successfully executed")
