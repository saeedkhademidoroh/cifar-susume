# Import standard libraries
import os
from pathlib import Path
import time
import json
from datetime import datetime, timezone
import shutil

# Import project-specific libraries
from config import CONFIG

# Function to log to json
def log_to_json(path, key, record=None, error=False):
    """
    Logs a dictionary to a JSON file.

    Args:
        path (Path or str): Directory where log should be written.
        key (str): Top-level key in result.json (ignored for error logs).
        record (dict): Dictionary of information to log.
        error (bool): If True, logs to a separate error_<timestamp>.json file.
    """

    # Print header for function execution
    print("\n🎯 log_to_json")

    # Create empty record if none is provided
    if record is None:
        record = {}

    # Ensure the directory exists
    os.makedirs(path, exist_ok=True)

    # Automatically attach timestamp metadata if missing
    if "timestamp" not in record:
        ts = time.time()
        record["timestamp"] = ts
        record["timestamp_utc"] = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()

    # Handle error logging mode
    if error:
        error_file = Path(path) / f"error_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json"
        with open(error_file, "w") as f:
            json.dump(record, f, indent=2)
        print(f"\n❌ Error: {error_file}")
        return

    # Handle standard result logging mode
    log_file = Path(path) / "result.json"

    # Load existing log file or start new
    if log_file.exists():
        with open(log_file, "r") as f:
            data = json.load(f)
    else:
        data = {}

    # Create new list for key if not already present
    if key not in data:
        data[key] = []

    # Append new record under key
    data[key].append(record)

    # Write updated log to disk
    with open(log_file, "w") as f:
        json.dump(data, f, indent=2)

    # Confirm logging action
    print(f"\n📝 Logged: key='{key}', file='{log_file.name}'")



# Function to clean output folders if CLEAN_MODE is enabled
def clean_old_outputs(flag=False):
    """
    Deletes standard output folders before starting experiments.
    This includes: RESULT, MODEL, LOG, ERROR, CHECKPOINT directories.
    Triggered only if flag is True.
    """
    print("\n🧹 Checking CLEAN_MODE setting...")

    if flag:
        print("\n🧼 CLEAN_MODE is ON — removing output folders...\n")
        targets = [
            CONFIG.RESULT_PATH,
            CONFIG.MODEL_PATH,
            CONFIG.LOG_PATH,
            CONFIG.ERROR_PATH,
            CONFIG.CHECKPOINT_PATH,
        ]
        for path in targets:
            if path.exists():
                print(f"🗑️  Removing: {path}")
                shutil.rmtree(path, ignore_errors=True)
            else:
                print(f"⚪ Skipped: {path} (not found)")
    else:
        print("\n🚫 CLEAN_MODE is OFF — skipping deletion.")


# Print confirmation message
print("\n✅ log.py successfully executed")
