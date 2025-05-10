# Import standard libraries
import json
from pathlib import Path
from dataclasses import dataclass


# Dataclass to represent an immutable config loaded from config.json
@dataclass(frozen=True)
class Config:
    """
    Immutable configuration class loaded from config.json.

    All attributes are loaded dynamically from keys in the config file.
    Paths in particular are resolved relative to the script's location.
    This object is imported as CONFIG and used globally across all modules.
    """

    # Paths and settings expected in config.json
    CONFIG_PATH: Path
    DATA_PATH: Path
    LOG_PATH: Path
    CHECKPOINT_PATH: Path
    RESULT_PATH: Path
    EPOCHS_COUNT: int
    BATCH_SIZE: int
    VALIDATION_SPLIT: float
    NUM_WORKERS: int
    CLEAN_OUTPUTS: bool

    @staticmethod
    def load_from_json() -> "Config":
        """
        Loads and validates configuration from config.json in the current directory.
        Ensures all keys exist and builds a populated Config instance.
        """
        # Print header for function
        print("\n🎯 load_from_json")

        # Resolve the base directory (same folder as this script)
        current_dir = Path(__file__).parent

        # Path to the actual config.json file
        raw_config_path = current_dir / "config.json"

        # Raise an error if config file doesn't exist
        if not raw_config_path.exists():
            raise FileNotFoundError(f"❌ Configuration file not found: {raw_config_path}")

        # Load and parse the config data
        with open(raw_config_path, "r") as f:
            config_data = json.load(f)

        # Define all required keys expected in the config.json
        required_keys = [
            "CONFIG_PATH",
            "DATA_PATH",
            "LOG_PATH",
            "CHECKPOINT_PATH",
            "RESULT_PATH",
            "EPOCHS_COUNT",
            "BATCH_SIZE",
            "VALIDATION_SPLIT",
            "NUM_WORKERS",
            "CLEAN_OUTPUTS"
        ]

        # Check for any missing keys
        missing = [key for key in required_keys if key not in config_data]
        if missing:
            raise ValueError(f"❌ Missing required config keys: {missing}")

        # Build and return the populated Config dataclass
        return Config(
            CONFIG_PATH=current_dir / config_data["CONFIG_PATH"],
            DATA_PATH=current_dir / config_data["DATA_PATH"],
            LOG_PATH=current_dir / config_data["LOG_PATH"],
            CHECKPOINT_PATH=current_dir / config_data["CHECKPOINT_PATH"],
            RESULT_PATH=current_dir / config_data["RESULT_PATH"],
            EPOCHS_COUNT=config_data["EPOCHS_COUNT"],
            BATCH_SIZE=config_data["BATCH_SIZE"],
            VALIDATION_SPLIT=config_data["VALIDATION_SPLIT"],
            NUM_WORKERS=config_data["NUM_WORKERS"],
            CLEAN_OUTPUTS=config_data["CLEAN_OUTPUTS"]
        )


# Load config at module level for global access
CONFIG = Config.load_from_json()

# Print confirmation
print("\n✅ config.py successfully executed")
