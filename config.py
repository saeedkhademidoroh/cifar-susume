# Import standard libraries
from dataclasses import dataclass
from pathlib import Path
import json


# Class for immutable configuration
@dataclass(frozen=True)
class Config:
    """
    Immutable configuration object for the CIFAR experiment system.

    Loaded from a config.json file, it contains:
    - Directory paths for data, logs, results, and checkpoints
    - Training hyperparameters
    - Execution mode flags
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

    # Function to load configuration from file
    @staticmethod
    def load_config(path: Path) -> "Config":
        """
        Function to load configuration from a JSON file.

        Loads and parses the specified config.json file, resolves its path,
        and constructs an immutable Config dataclass.

        Args:
            path (Path): Full path to the JSON configuration file.

        Returns:
            Config: An initialized and validated Config dataclass instance.
        """

        # Print header for function execution
        print("\n🎯 load_config")

        # Announce which file is being loaded
        print(f"\n📂 Loading custom configuration:\n{path}")

        # Get the parent directory of the config path
        base_path = path.parent

        # Read the config JSON into a dictionary
        with open(path, "r") as f:
            config_data = json.load(f)

        # Return validated and resolved Config object
        return Config._from_dict(config_data, base_path)  # Return initialized config object



    # Function to initialize Config object from dictionary
    @staticmethod
    def _from_dict(config_data: dict, base_path: Path) -> "Config":
        """
        Function to build a Config instance from a dictionary.

        Args:
            config_data (dict): Parsed JSON dictionary
            base_path (Path): Base directory for resolving relative paths

        Returns:
            Config: Fully populated configuration object
        """

        # Define required keys for validation
        required_keys = [
            "CONFIG_PATH",
            "DATA_PATH",
            "LOG_PATH",
            "CHECKPOINT_PATH",
            "RESULT_PATH",
            "MODEL_PATH",
            "ERROR_PATH",
            "EPOCHS_COUNT",
            "BATCH_SIZE",
            "LIGHT_MODE"
        ]

        # Validate all required keys are present
        missing = [key for key in required_keys if key not in config_data]
        if missing:
            raise ValueError(f"❌ ValueError:\nmissing={missing}\n")

        # Resolve all paths relative to module location
        root_path = Path(__file__).parent

        # Return immutable configuration object
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


# Print module successfully executed
print("\n✅ config.py successfully executed")
