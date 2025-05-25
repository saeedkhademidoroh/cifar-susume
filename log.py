# Import standard libraries
import os
from pathlib import Path
import time
import json
from datetime import datetime, timezone
import shutil

# Import project-specific libraries
from config import CONFIG


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
    print("\n🎯  log_to_json")

    # Create empty record if not provided
    if record is None:
        record = {}

    # Ensure target directory exists
    os.makedirs(path, exist_ok=True)

    # Auto-append timestamps if missing
    if "timestamp" not in record:
        ts = time.time()
        record["timestamp"] = ts
        record["timestamp_utc"] = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()

    # Handle error logging mode
    if error:
        error_file = Path(path) / f"error_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json"
        with open(error_file, "w") as f:
            json.dump(record, f, indent=2)
        print(f"\n❌ Error from log.py at log_to_json():\n{error_file}\n")
        return

    # Standard result logging mode
    log_file = Path(path) / "result.json"

    # Load or initialize result file
    if log_file.exists():
        with open(log_file, "r") as f:
            data = json.load(f)
    else:
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
    print(f"\n📝 Logging experiment result: key='{key}', file='{log_file.name}'")


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
    print("\n🎯  clean_old_output")

    if flag:
        print("\n🧼  CLEAN_MODE is ON - Cleaning old output directories")
        targets = [
            CONFIG.MODEL_PATH,
            CONFIG.ERROR_PATH,
            CONFIG.CHECKPOINT_PATH,
        ]
        print(f"\n🗑️   Cleaning old experiment output")
        for path in targets:
            if path.exists():
                print(f"\n{path}")
                shutil.rmtree(path, ignore_errors=True)
            else:
                print(f"❌  Failing to clean old output:\n{path}")
    else:
        print("\n🚫  CLEAN_MODE is OFF — skipping old output directories")


# Print module successfully executed
print("\n✅  log.py successfully executed")
