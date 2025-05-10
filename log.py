# Import standard libraries
import os
import time
import json
from datetime import datetime, timezone


# Function to ensure output directory exists
def ensure_dir(path):
    """
    Creates the parent directory of the given path if it doesn't already exist.
    """

    # Print header for function execution
    print("\n🎯 ensure_dir")

    os.makedirs(os.path.dirname(path), exist_ok=True)



# Function to log a dictionary under a named key in JSON
def log_to_json(file_path, key, record=None):
    """
    Appends a record to a JSON log file under the specified key.
    Adds both POSIX and ISO UTC timestamps if not already present.

    Args:
        file_path (Path or str): JSON log file path (defaults to LOG_PATH)
        key (str): Key under which to store the log record
        record (dict): Dictionary to log
    """

    # Print header for function execution
    print("\n🎯 log_to_json")

    # Initialize record as empty if not provided
    if record is None:
        record = {}

    # Ensure parent directory for log exists
    ensure_dir(file_path)

    # Load existing log file if it exists; otherwise start a new structure
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            data = json.load(f)
    else:
        data = {}

    # Inject timestamps if not already present in the record
    if "timestamp" not in record:
        timestamp = time.time()
        record["timestamp"] = timestamp
        record["timestamp_utc"] = datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat()

    # Initialize key if this is the first entry for it
    if key not in data:
        data[key] = []

    # Append current record to the list under the specified key
    data[key].append(record)

    # Overwrite the JSON file with updated content
    with open(file_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"\n📝 Saved to result JSON: key='{key}', file='{file_path.name}'")



# Print confirmation message
print("\n✅ log.py successfully executed")
