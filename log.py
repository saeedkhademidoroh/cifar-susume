# Import standard libraries
import os
from pathlib import Path
import time
import json
from datetime import datetime, timezone


# Function to log a dictionary under a named key in a JSON file
def log_to_json(path, key, record=None, error=False):
    """
    Logs a record to a JSON file. For normal logs, appends under 'key' inside result.json.
    For errors, creates a separate error_<timestamp>.json file in ERROR_PATH.

    Args:
        path (Path or str): Target directory.
        key (str): Key under which to store the record (ignored for error logs).
        record (dict): Record dictionary to log.
        error (bool): Whether to treat this as an error log (default: False).
    """

    # Print header for function execution
    print("\n🎯 log_to_json")

    if record is None:
        record = {}

    # Ensure path exists
    os.makedirs(path, exist_ok=True)

    # Inject timestamp if not already present
    if "timestamp" not in record:
        ts = time.time()
        record["timestamp"] = ts
        record["timestamp_utc"] = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()

    # Handle error case
    if error:
        error_file = path / f"error_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json"
        with open(error_file, "w") as f:
            json.dump(record, f, indent=2)
        print(f"\n❌ Error: {error_file}")
        return

    # Standard result log
    log_file = Path(path) / "result.json"
    if log_file.exists():
        with open(log_file, "r") as f:
            data = json.load(f)
    else:
        data = {}

    if key not in data:
        data[key] = []

    data[key].append(record)

    with open(log_file, "w") as f:
        json.dump(data, f, indent=2)

    print(f"\n📝 Logged: key='{key}', file='{log_file.name}'")


# Print confirmation message
print("\n✅ log.py successfully executed")
