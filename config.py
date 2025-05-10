# Standard imports
import json
from pathlib import Path
from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    """
    Immutable configuration class loaded from config.json.

    All attributes are loaded dynamically based on keys in the config file.
    Paths in particular are resolved relative to the script's location.

    """

    # Generic fields that will be set dynamically from JSON
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
        Loads and validates configuration from artifact/json/config.json.
        Ensures all keys exist and resolves path values relative to script directory.

        Returns:
            Config: Frozen dataclass instance with resolved values.

        Raises:
            FileNotFoundError: If config.json is missing.
            ValueError: If required keys are missing.
        """
        current_dir = Path(__file__).parent
        raw_config_path = current_dir / "config.json"

        if not raw_config_path.exists():
            raise FileNotFoundError(f"❌ Configuration file not found: {raw_config_path}")

        with open(raw_config_path, "r") as f:
            config_data = json.load(f)

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
        missing = [key for key in required_keys if key not in config_data]
        if missing:
            raise ValueError(f"❌ Missing required config keys: {missing}")

        return Config(
            CONFIG_PATH=current_dir / config_data["CONFIG_PATH"],
            DATA_PATH=current_dir / config_data["DATA_PATH"],
            LOG_PATH=current_dir / config_data["LOG_PATH"],
            CHECKPOINT_PATH=current_dir / config_data["CHECKPOINT_PATH"],
            RESULT_PATH = current_dir / config_data["RESULT_PATH"],
            EPOCHS_COUNT=config_data["EPOCHS_COUNT"],
            BATCH_SIZE=config_data["BATCH_SIZE"],
            VALIDATION_SPLIT=config_data["VALIDATION_SPLIT"],
            NUM_WORKERS=config_data["NUM_WORKERS"],
            CLEAN_OUTPUTS = config_data["CLEAN_OUTPUTS"]
        )


# Load config at module level
CONFIG = Config.load_from_json()

# Confirmation
print("\n✅ config.py successfully executed")
