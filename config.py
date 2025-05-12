# Import standard libraries
import json
from pathlib import Path
from dataclasses import dataclass


# Dataclass for immutable config
from dataclasses import dataclass
from pathlib import Path
import json


@dataclass(frozen=True)
class Config:
    """
    Immutable configuration object for the CIFAR experiment system.

    This dataclass is loaded from config.json at runtime and contains:
    - Paths for data, logs, results, and checkpoints
    - Training hyperparameters
    - Flags to control cleaning behavior and runtime mode
    """

    # Path to the config.json
    CONFIG_PATH: Path

    # Path to dataset/
    DATA_PATH: Path

    # Path to log/
    LOG_PATH: Path

    # Path to checkpoint/
    CHECKPOINT_PATH: Path

    # Path to result/
    RESULT_PATH: Path

    # Number of training epochs
    EPOCHS_COUNT: int

    # Number of batch size
    BATCH_SIZE: int

    # Flag for clearing log/
    CLEAN_LOG: bool

    # Flag for clearing checkpoint/
    CLEAN_CHECKPOINT: bool

    # Flag for clearing result/
    CLEAN_RESULT: bool

    # Flag for lightweight mode
    LIGHT_MODE: bool

    @staticmethod
    def load_from_json() -> "Config":
        """
        Loads and validates configuration from config.json located
        in the same directory as this script.

        Returns:
            Config: A fully populated, immutable Config object.

        Raises:
            FileNotFoundError: If the config.json file is missing.
            ValueError: If any required keys are missing from the file.
        """
        print("\n🎯 load_from_json")
        current_dir = Path(__file__).parent
        raw_config_path = current_dir / "config.json"

        if not raw_config_path.exists():
            raise FileNotFoundError(f"❌ FileNotFoundError:\nraw_config_path={raw_config_path}\n")

        with open(raw_config_path, "r") as f:
            config_data = json.load(f)

        # Required keys that must be present in config.json
        required_keys = [
            "CONFIG_PATH",
            "DATA_PATH",
            "LOG_PATH",
            "CHECKPOINT_PATH",
            "RESULT_PATH",
            "EPOCHS_COUNT",
            "BATCH_SIZE",
            "CLEAN_LOG",
            "CLEAN_CHECKPOINT",
            "CLEAN_RESULT",
            "LIGHT_MODE"
        ]

        # Validate presence of required keys
        missing = [key for key in required_keys if key not in config_data]
        if missing:
            raise ValueError(f"❌ ValueError:\nmissing={missing}\n")

        # Create and return frozen Config object
        return Config(
            CONFIG_PATH=current_dir / config_data["CONFIG_PATH"],
            DATA_PATH=current_dir / config_data["DATA_PATH"],
            LOG_PATH=current_dir / config_data["LOG_PATH"],
            CHECKPOINT_PATH=current_dir / config_data["CHECKPOINT_PATH"],
            RESULT_PATH=current_dir / config_data["RESULT_PATH"],
            EPOCHS_COUNT=config_data["EPOCHS_COUNT"],
            BATCH_SIZE=config_data["BATCH_SIZE"],
            CLEAN_LOG=config_data["CLEAN_LOG"],
            CLEAN_CHECKPOINT=config_data["CLEAN_CHECKPOINT"],
            CLEAN_RESULT=config_data["CLEAN_RESULT"],
            LIGHT_MODE=config_data["LIGHT_MODE"]
        )


# Load the configurations from config.json
CONFIG = Config.load_from_json()

# Print confirmation message
print("\n✅ config.py successfully executed")
