# Import standard libraries
import json
from pathlib import Path
from dataclasses import dataclass


# Dataclass for immutable config
@dataclass(frozen=True)
class Config:
    """
    Immutable configuration object for the CIFAR experiment system.

    This dataclass is loaded from default.json at runtime and contains:
    - Paths for data, logs, results, and checkpoints
    - Training hyperparameters
    - Flags to control cleaning behavior and runtime mode
    """

    CONFIG_PATH: Path
    DATA_PATH: Path
    LOG_PATH: Path
    CHECKPOINT_PATH: Path
    RESULT_PATH: Path
    MODEL_PATH: Path
    HISTORY_PATH: Path
    ERROR_PATH: Path
    EPOCHS_COUNT: int
    BATCH_SIZE: int
    CLEAN_LOG: bool
    CLEAN_CHECKPOINT: bool
    CLEAN_RESULT: bool
    CLEAN_MODEL: bool
    CLEAN_HISTORY: bool
    CLEAN_ERROR: bool
    LIGHT_MODE: bool

    # Function to load the default configuration from artifact/config/default.json
    @staticmethod
    def load_default_config() -> "Config":

        # Print header for function execution
        print("\n🎯 load_default_config")

        # Define default config path and display it
        current_dir = Path(__file__).parent
        raw_config_path = current_dir / "artifact/config/default.json"
        print(f"\n📂 Loading default config:\n{raw_config_path}")

        # Check that the file exists, raise an error if not
        if not raw_config_path.exists():
            raise FileNotFoundError(f"❌ FileNotFoundError:\nraw_config_path={raw_config_path}\n")

        # Load and parse the JSON config file
        with open(raw_config_path, "r") as f:
            config_data = json.load(f)

        # Validate required keys
        required_keys = [
            "CONFIG_PATH", "DATA_PATH", "LOG_PATH", "CHECKPOINT_PATH", "RESULT_PATH", "MODEL_PATH",
            "HISTORY_PATH", "ERROR_PATH",
            "EPOCHS_COUNT", "BATCH_SIZE",
            "CLEAN_LOG", "CLEAN_CHECKPOINT", "CLEAN_RESULT", "CLEAN_MODEL", "CLEAN_HISTORY", "CLEAN_ERROR", "LIGHT_MODE"
        ]
        missing = [key for key in required_keys if key not in config_data]
        if missing:
            raise ValueError(f"❌ ValueError:\nmissing={missing}\n")

        return Config._from_dict(config_data, current_dir)

    # Function to load an experiment-specific config from custom path
    @staticmethod
    def load_from_custom_path(custom_path: Path) -> "Config":
        """
        Loads configuration from a user-defined .json file (e.g. colab.json, desktop.json)
        """
        print("\n🎯 load_from_custom_path")
        print(f"\n📂 Loading custom configuration:\n{custom_path}")
        base_path = custom_path.parent
        with open(custom_path, "r") as f:
            config_data = json.load(f)
        return Config._from_dict(config_data, base_path)

    # Shared constructor from dictionary data + base path
    @staticmethod
    def _from_dict(config_data: dict, base_path: Path) -> "Config":
        """
        Internal shared method to construct the Config object from a dictionary and base path.
        """

        # Ensure all required fields are present
        required_keys = [
            "CONFIG_PATH", "DATA_PATH", "LOG_PATH", "CHECKPOINT_PATH", "RESULT_PATH", "MODEL_PATH",
            "HISTORY_PATH", "ERROR_PATH",
            "EPOCHS_COUNT", "BATCH_SIZE",
            "CLEAN_LOG", "CLEAN_CHECKPOINT", "CLEAN_RESULT", "CLEAN_MODEL", "CLEAN_HISTORY", "CLEAN_ERROR", "LIGHT_MODE"
        ]
        missing = [key for key in required_keys if key not in config_data]
        if missing:
            raise ValueError(f"❌ ValueError:\nmissing={missing}\n")

        # Return populated frozen dataclass
        return Config(
            CONFIG_PATH=base_path / config_data["CONFIG_PATH"],
            DATA_PATH=base_path / config_data["DATA_PATH"],
            LOG_PATH=base_path / config_data["LOG_PATH"],
            CHECKPOINT_PATH=base_path / config_data["CHECKPOINT_PATH"],
            RESULT_PATH=base_path / config_data["RESULT_PATH"],
            MODEL_PATH=base_path / config_data["MODEL_PATH"],
            HISTORY_PATH=base_path / config_data["HISTORY_PATH"],
            ERROR_PATH=base_path / config_data["ERROR_PATH"],
            EPOCHS_COUNT=config_data["EPOCHS_COUNT"],
            BATCH_SIZE=config_data["BATCH_SIZE"],
            CLEAN_LOG=config_data["CLEAN_LOG"],
            CLEAN_CHECKPOINT=config_data["CLEAN_CHECKPOINT"],
            CLEAN_RESULT=config_data["CLEAN_RESULT"],
            CLEAN_MODEL=config_data["CLEAN_MODEL"],
            CLEAN_HISTORY=config_data["CLEAN_HISTORY"],
            CLEAN_ERROR=config_data["CLEAN_ERROR"],
            LIGHT_MODE=config_data["LIGHT_MODE"]
        )


# Load the configurations from default.json
CONFIG = Config.load_default_config()

# Print confirmation message
print("\n✅ config.py successfully executed")
