# Import standard libraries
import os
from pathlib import Path
import time
import json
from datetime import datetime, timezone


# Function to log a dictionary under a named key in JSON
def log_to_json(path, key, record=None):
    """
    Appends a record to a JSON log file under the specified key.
    Adds both POSIX and ISO UTC timestamps if not already present.

    Args:
        path (Path or str): Directory where the log file should reside.
        key (str): Key under which to store the log record.
        record (dict): Dictionary to log.
    """

    print("\n🎯 log_to_json")

    if record is None:
        record = {}

    # Ensure parent directory exists
    os.makedirs(path, exist_ok=True)

    # Define full file path
    log_file = Path(path) / "result.json"

    # Load or initialize log content
    if log_file.exists():
        with open(log_file, "r") as f:
            data = json.load(f)
    else:
        data = {}

    # Inject timestamps
    if "timestamp" not in record:
        timestamp = time.time()
        record["timestamp"] = timestamp
        record["timestamp_utc"] = datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat()

    if key not in data:
        data[key] = []

    data[key].append(record)

    with open(log_file, "w") as f:
        json.dump(data, f, indent=2)

    print(f"\n📝 Logged: key='{key}', file='{log_file.name}'")



# Print confirmation message
print("\n✅ log.py successfully executed")
