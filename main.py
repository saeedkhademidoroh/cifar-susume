from pathlib import Path
from config import CONFIG
from experiment import run_experiment

# Print confirmation message
print("\n✅ main.py is being executed")

# Disable GPU (force CPU usage)
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Resolve base config path from CONFIG object
CONFIG_DIR = CONFIG.CONFIG_PATH

# Build config_map using CONFIG_PATH + filenames
config_map = {
    model: {
        1: CONFIG_DIR / "laptop.json",
    } for model in range(5)
}

# Run all models using laptop config for both runs
run_experiment((0, 4))