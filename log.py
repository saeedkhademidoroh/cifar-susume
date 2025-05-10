# Standard libraries
import os
import time
import json
from datetime import datetime, timezone


# Ensure output directory exists
def ensure_dir(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)

# General-purpose JSON logger
def log_to_json(file_path, key, record=None):
    """
    Appends a record to a JSON log file under the specified key.
    Adds both POSIX and ISO UTC timestamps if not already present.

    Args:
        file_path (Path or str): JSON log file path (defaults to LOG_PATH)
        key (str): Key under which to store the log record
        record (dict): Dictionary to log
    """
    if record is None:
        record = {}

    ensure_dir(file_path)

    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            data = json.load(f)
    else:
        data = {}

    if "timestamp" not in record:
        timestamp = time.time()
        record["timestamp"] = timestamp
        record["timestamp_utc"] = datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat()

    if key not in data:
        data[key] = []

    data[key].append(record)

    with open(file_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"📝 Saved to result JSON: key='{key}', file='{file_path.name}'")

# Print confirmation message
print("\n✅ log.py successfully executed")
