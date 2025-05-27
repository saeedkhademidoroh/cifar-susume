# Import standard libraries
import os
from pathlib import Path
import time
import json
from datetime import datetime, timezone, timedelta
import shutil

# Import project-specific libraries
from config import CONFIG


# Function to clean old output
def clean_old_output(flag=False):
    """
    Cleans model, checkpoint, and error directories if CLEAN_MODE is enabled.

    Primarily used to remove previous run artifacts before starting a new experiment.

    Args:
        flag (bool): If True, triggers cleanup. Otherwise, does nothing.

    Returns:
        None
    """

    # Print header for function execution
    print("\n🎯  clean_old_output is executing ...")

    if flag:
        print("\n🧼  CLEAN_MODE is ON - Cleaning old output directories ...")
        targets = [
            CONFIG.MODEL_PATH,
            CONFIG.ERROR_PATH,
            CONFIG.CHECKPOINT_PATH,
        ]
        print(f"\n🧹  Cleaning old experiment output directories:")
        for path in targets:
            if path.exists():
                print(f"→ {path}")
                shutil.rmtree(path, ignore_errors=True)
            else:
                print(f"→ {path}")
    else:
        print("\n🧼  CLEAN_MODE is OFF — skipping old output directories ...")


# Function to log structured data to JSON
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


    # Print header for function execution
    print("\n🎯  log_to_json is executing ...")

    # Create empty record if not provided
    if record is None:
        record = {}

    # Ensure target directory exists
    os.makedirs(path, exist_ok=True)

    # Auto-append timestamps if missing
    if "timestamp" not in record:
        ts = time.time()

        # Store raw POSIX timestamp
        record["timestamp"] = ts

        # Store UTC-formatted timestamp
        record["timestamp_utc"] = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()

        # Store Tehran-formatted timestamp (+03:30)
        tehran_offset = timezone(timedelta(hours=3, minutes=30))
        record["timestamp_tehran"] = datetime.fromtimestamp(ts, tz=tehran_offset).isoformat()

    # Handle error logging mode
    if error:
        # Construct error file path with timestamped name
        error_file = Path(path) / f"error_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json"

        # Write the error record to the JSON file
        with open(error_file, "w") as f:
            json.dump(record, f, indent=2)

        # Print confirmation message with file path
        print(f"\n\n❌  Error from utility.py at log_to_json()!\n{error_file}\n\n")

        # Exit early since this is error mode — no need to append to result.json
        return

    # Standard result logging mode
    log_file = Path(path) / "result.json"

    # Load existing result.json if it exists
    if log_file.exists():
        with open(log_file, "r") as f:
            data = json.load(f)
    else:
        # Initialize empty structure if result.json is missing
        data = {}

    # Initialize log list under key if needed
    if key not in data:
        data[key] = []

    # Append new record
    data[key].append(record)

    # Save updated log
    with open(log_file, "w") as f:
        json.dump(data, f, indent=2)

    # Confirm successful logging
    print(f"\n📝  Logging experiment result ...\n→ Key: '{key}'\n→ File: '{log_file.name}'")


# Function to extract history metrics
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

    # Print header for function execution
    print("\n🎯  extract_history_metrics is executing ...")

    # Convert to raw dict if wrapped in a History object
    history = history.history if hasattr(history, "history") else history

    # Extract training metrics
    train_loss = history.get("loss", [])
    train_acc = history.get("accuracy", [])

    # Log fallback if empty
    if not train_loss or not train_acc:
        print("\n⚠️  Training history is incomplete — loss/accuracy may be missing")

    # Extract core training metrics (loss and accuracy)
    metrics = {
        "min_train_loss": min(train_loss) if train_loss else None,  # Best (lowest) training loss
        "min_train_loss_epoch": train_loss.index(min(train_loss)) + 1 if train_loss else None,  # Epoch of best training loss
        "max_train_acc": max(train_acc) if train_acc else None,  # Best (highest) training accuracy
        "max_train_acc_epoch": train_acc.index(max(train_acc)) + 1 if train_acc else None,  # Epoch of best training accuracy
    }

    # Extract optional validation metrics (may be missing in light mode)
    val_loss = history.get("val_loss", [])
    val_acc = history.get("val_accuracy", [])

    metrics["min_val_loss"] = min(val_loss) if val_loss else None  # Best validation loss
    metrics["min_val_loss_epoch"] = val_loss.index(min(val_loss)) + 1 if val_loss else None  # Epoch of best val loss
    metrics["max_val_acc"] = max(val_acc) if val_acc else None  # Best validation accuracy
    metrics["max_val_acc_epoch"] = val_acc.index(max(val_acc)) + 1 if val_acc else None  # Epoch of best val accuracy

    # Print extracted key stats
    print(f"\n📈  Training metrics have been extracted:")
    print(f"→ Min train loss: {metrics['min_train_loss']} at epoch {metrics['min_train_loss_epoch']}")
    print(f"→ Max train acc:  {metrics['max_train_acc']} at epoch {metrics['max_train_acc_epoch']}")
    if val_loss:
        print(f"→ Min val loss:  {metrics['min_val_loss']} at epoch {metrics['min_val_loss_epoch']}")
    if val_acc:
        print(f"→ Max val acc:   {metrics['max_val_acc']} at epoch {metrics['max_val_acc_epoch']}")

    # Return dictionary containing all extracted min/max metrics
    return metrics


# Print module successfully executed
print("\n✅  utility.py successfully executed.")
