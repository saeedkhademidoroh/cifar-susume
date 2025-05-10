# Standard imports
import json
from pathlib import Path
from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    """
    Immutable configuration class loaded from config.json.

    All attributes are loaded dynamically based on keys in the config file.
    Paths are resolved relative to the script's location.

    Example keys (from config.json):
        - CONFIG_PATH
        - DATA_PATH
        - RESULTS_PATH
        - BATCH_SIZE
        - NUM_WORKERS
    """

    # Generic fields that will be set dynamically from JSON
    CONFIG_PATH: Path
    DATA_PATH: Path
    RESULTS_PATH: Path
    BATCH_SIZE: int
    NUM_WORKERS: int

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
        raw_config_path = current_dir / "artifact/json/config.json"

        if not raw_config_path.exists():
            raise FileNotFoundError(f"❌ Configuration file not found: {raw_config_path}")

        with open(raw_config_path, "r") as f:
            config_data = json.load(f)

        required_keys = [
            "CONFIG_PATH",
            "DATA_PATH",
            "RESULTS_PATH",
            "BATCH_SIZE",
            "NUM_WORKERS"
        ]
        missing = [key for key in required_keys if key not in config_data]
        if missing:
            raise ValueError(f"❌ Missing required config keys: {missing}")

        return Config(
            CONFIG_PATH=current_dir / config_data["CONFIG_PATH"],
            DATA_PATH=current_dir / config_data["DATA_PATH"],
            RESULTS_PATH=current_dir / config_data["RESULTS_PATH"],
            BATCH_SIZE=config_data["BATCH_SIZE"],
            NUM_WORKERS=config_data["NUM_WORKERS"]
        )


# Load config at module level
CONFIG = Config.load_from_json()

# Confirmation
print("\n✅ config.py successfully executed")
