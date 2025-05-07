# Import standard libraries
import json
from pathlib import Path

# Base directory for this script
BASE_DIR = Path(__file__).parent

# Path to config.json (only hardcoded reference)
CONFIG_PATH = BASE_DIR / "artifact/json/config.json"

# Load configuration from JSON file
def load_config(path=CONFIG_PATH):
    if not path.exists():
        raise FileNotFoundError(f"❌ Config file not found: {path}")
    with open(path, "r") as f:
        config = json.load(f)
    print(f"✅ Loaded config from {path}")
    return config

# Load once globally
CONFIG = load_config()

# Resolve key paths dynamically relative to BASE_DIR
DATA_DIR = BASE_DIR / CONFIG["data_dir"]
LOG_PATH = BASE_DIR / CONFIG["log_path"]
CONFIG_PATH_RESOLVED = BASE_DIR / CONFIG["config_path"]

# Other config values
BATCH_SIZE = CONFIG["batch_size"]
NUM_WORKERS = CONFIG["num_workers"]
