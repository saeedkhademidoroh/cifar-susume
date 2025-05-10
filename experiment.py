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


# Function to remove output folders (log, result, checkpoint)
def remove_output(flag=CONFIG.CLEAN_OUTPUTS):
    """
    Removes the entire checkpoint, log, and result directories if enabled.

    Parameters:
        flag (bool): If True, deletes each target directory. If False, does nothing.
    """

    # Print header for function execution
    print("\n🎯 remove_output\n")

    if not flag:
        print("🗑️  Skipped output folder removal\n")
        return

    target_folders = [
        CONFIG.LOG_PATH,
        CONFIG.RESULT_PATH,
        CONFIG.CHECKPOINT_PATH,
    ]

    for folder in target_folders:
        if folder.exists():
            try:
                shutil.rmtree(folder)
                print(f"🗑️  Successfully removed entire folder:\n {folder}\n")
            except Exception as e:
                print(f"⚠️ Failed to remove folder:\n {folder} {e}\n")
        else:
            raise FileNotFoundError(f"❌ Target folder not found:\n {folder}\n")



# Function to collect metadata about a trained model
def collect_experiment_result(model, history, model_name: str):
    """
    Extracts model metrics and structure for experiment summary.

    Args:
        model (tf.keras.Model): The trained model.
        history (keras.callbacks.History): Training history returned by fit().
        model_name (str): Identifier for the model used (e.g., m1, m2).

    Returns:
        dict: Dictionary containing training metadata for the experiment.
    """

    # Print header for function execution
    print("\n🎯 collect_experiment_result")

    time = datetime.datetime.now().strftime("%H:%M:%S")
    layers_count = len(model.layers)
    optimizer = type(model.optimizer).__name__
    val_acc = max(history.history.get("val_accuracy", [0]))
    loss = min(history.history.get("loss", [0]))

    return {
        "model": model_name,
        "time": time,
        "layers": layers_count,
        "optimizer": optimizer,
        "val_accuracy": round(val_acc, 4),
        "min_loss": round(loss, 4)
    }



# Function to run training experiments and log results
def run_experiment(model_numbers, runs=1):
    """
    Executes training experiments and logs all stdout/stderr to a timestamped log file.

    Args:
        model_numbers (int | tuple): Single model number (int) or a range (tuple of ints).
        runs (int): Number of times to train each selected model.
    """

    # Print header for function execution
    print("\n🎯 run_experiment")

    # Clean previous outputs
    remove_output()

    # Create timestamp for log and result file names
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Define log file path
    log_file = CONFIG.LOG_PATH / f"log_{timestamp}.txt"
    CONFIG.LOG_PATH.mkdir(parents=True, exist_ok=True)

    # Redirect stdout/stderr to log file
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    f = open(log_file, "a")
    sys.stdout = f
    sys.stderr = f

    try:
        # Load dataset
        train_data, train_labels, _, _ = load_dataset(one_hot=True)

        # Handle different model selection inputs
        if isinstance(model_numbers, int):
            model_numbers = [model_numbers]  # Single model
        elif isinstance(model_numbers, tuple) and len(model_numbers) == 2:
            model_numbers = list(range(model_numbers[0], model_numbers[1] + 1))  # Range of models

        # Prepare JSON result file
        CONFIG.RESULT_PATH.mkdir(parents=True, exist_ok=True)
        result_file_path = CONFIG.RESULT_PATH / f"result_{timestamp}.json"
        all_result = []

        # Loop over each model and run specified number of times
        for model_number in model_numbers:
            for run in range(1, runs + 1):
                print(f"\n🚀 Launching m{model_number} ({run}/{runs}) ...")

                # Build model
                model, description = build_model(model_number)

                # Train model
                trained_model, history = train_model(
                    train_data, train_labels,
                    model,
                    model_name=f"m{model_number}",
                    verbose=0,
                    result_file_path=result_file_path
                )

                # Summarize result
                result = collect_experiment_result(trained_model, history, model_name=f"m{model_number}")
                all_result.append(result)

        # Write results to result JSON
        with open(result_file_path, "w") as jf:
            json.dump(all_result, jf, indent=2)

    finally:
        # Restore normal stdout/stderr
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        f.close()


# Print confirmation message
print("\n✅ experiment.py successfully executed")
