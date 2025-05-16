# Import standard libraries
from dataclasses import dataclass
from pathlib import Path
import json


# Dataclass for immutable config
@dataclass(frozen=True)
class Config:
    """
    Immutable configuration object for the CIFAR experiment system.

    This dataclass is loaded from a config.json file at runtime and contains:
    - Paths for data, logs, results, and checkpoints
    - Training hyperparameters
    - Flags to control runtime mode
    """

    CONFIG_PATH: Path
    DATA_PATH: Path
    LOG_PATH: Path
    CHECKPOINT_PATH: Path
    RESULT_PATH: Path
    MODEL_PATH: Path
    ERROR_PATH: Path
    EPOCHS_COUNT: int
    BATCH_SIZE: int
    LIGHT_MODE: bool

    # Function to load configuration file
    @staticmethod
    def load_config(path: Path) -> "Config":
        """
        Loads configuration from a user-defined .json file
        """

        # Print header for function execution
        print("\n🎯 load_config")


        print(f"\n📂 Loading custom configuration:\n{path}")
        base_path = path.parent
        with open(path, "r") as f:
            config_data = json.load(f)
        return Config._from_dict(config_data, base_path)


    # Function to construct config object
    @staticmethod
    def _from_dict(config_data: dict, base_path: Path) -> "Config":
        """
        Internal shared method to construct the Config object from a dictionary and base path.
        """

        # Validate required keys
        required_keys = [

            # Paths
            "CONFIG_PATH",
            "DATA_PATH",
            "LOG_PATH",
            "CHECKPOINT_PATH",
            "RESULT_PATH",
            "MODEL_PATH",
            "ERROR_PATH",

            # Parameters
            "EPOCHS_COUNT",
            "BATCH_SIZE",

            # Modes
            "LIGHT_MODE"
        ]

        # Check for missing keys
        missing = [key for key in required_keys if key not in config_data]
        if missing:
            raise ValueError(f"❌ ValueError:\nmissing={missing}\n")

        # Resolve all paths relative to the project root
        root_path = Path(__file__).parent

        # Return populated frozen dataclass
        return Config(
            CONFIG_PATH=root_path / config_data["CONFIG_PATH"],
            DATA_PATH=root_path / config_data["DATA_PATH"],
            LOG_PATH=root_path / config_data["LOG_PATH"],
            CHECKPOINT_PATH=root_path / config_data["CHECKPOINT_PATH"],
            RESULT_PATH=root_path / config_data["RESULT_PATH"],
            MODEL_PATH=root_path / config_data["MODEL_PATH"],
            ERROR_PATH=root_path / config_data["ERROR_PATH"],
            EPOCHS_COUNT=config_data["EPOCHS_COUNT"],
            BATCH_SIZE=config_data["BATCH_SIZE"],
            LIGHT_MODE=config_data["LIGHT_MODE"]
        )


# Load default configuration from artifact/config/default.json
default_path = Path(__file__).parent / "artifact/config/default.json"
CONFIG = Config.load_config(default_path)

# Print module successfuly executed
print("\n✅ config.py successfully executed")
