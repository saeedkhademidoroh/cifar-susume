from pathlib import Path
from config import CONFIG
from experiment import run_experiment

# Print confirmation message
print("\n✅ main.py is being executed")

# Disable GPU (force CPU usage)
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Resolve base config path
config_path = CONFIG.CONFIG_PATH

# Explicit config_map
config_map = {
    0: {
        1: config_path / "clean.json"
    },
    1: {
        1: config_path / "desktop.json"
    },
    2: {
        1: config_path / "desktop.json"
    },
    3: {
        1: config_path / "desktop.json"
    },
    4: {
        1: config_path / "desktop.json"
    }
}

# Run only the specified models with their declared runs and configs
run_experiment(model_numbers=list(config_map.keys()), config_map=config_map)
