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
    - Data loading and augmentation flags
    - Model-level regularization flags
    - Optimizer parameters
    - Learning rate scheduling and early stopping
    - Training hyperparameters
    """

    CONFIG_PATH: Path              # Path to the folder containing config files
    DATA_PATH: Path                # Path to dataset root
    LOG_PATH: Path                 # Path to log output directory
    CHECKPOINT_PATH: Path          # Path to store model checkpoints
    RESULT_PATH: Path              # Path for final result.json outputs
    MODEL_PATH: Path               # Path to store serialized models
    ERROR_PATH: Path               # Path to store structured error logs

    LIGHT_MODE: bool               # Enable reduced dataset (e.g. 5k train, 1k test) for fast iteration
    AUGMENT_MODE: dict             # Augmentation config with subkeys: random_crop, random_flip, cutout

    L2_MODE: dict                  # L2 regularization config (enabled + lambda)
    DROPOUT_MODE: dict             # Dropout config (enabled + dropout rate, often disabled for ResNet)
    OPTIMIZER: dict                # Optimizer settings: type, learning rate, momentum
    SCHEDULE_MODE: dict            # LR scheduler config (manual StepLR + optional warmup)
    EARLY_STOP_MODE: dict          # Early stopping config (usually disabled for 200-epoch ResNet runs)
    AVERAGE_MODE: dict             # Model weight averaging config (SWA-style, enabled near end of training)
    TTA_MODE: dict                 # Test-time augmentation config (enabled + runs count)

    EPOCHS_COUNT: int              # Total number of training epochs (e.g. 200)
    BATCH_SIZE: int                # Training batch size (e.g. 128)

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
        print("\n🎯  load_config")

        # Announce which file is being loaded
        print(f"\n📂  Loading configuration file:\n{path}")

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
            "LIGHT_MODE",
            "AUGMENT_MODE",
            "L2_MODE",
            "DROPOUT_MODE",
            "OPTIMIZER",
            "SCHEDULE_MODE",
            "EARLY_STOP_MODE",
            "AVERAGE_MODE",
            "TTA_MODE",
            "EPOCHS_COUNT",
            "BATCH_SIZE"
        ]

        # Validate all required keys are present
        missing = [key for key in required_keys if key not in config_data]
        if missing:
            raise ValueError(f"❌ ValueError from config.py at _from_dict():\nmissing={missing}\n")

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
            LIGHT_MODE=config_data["LIGHT_MODE"],
            AUGMENT_MODE=config_data["AUGMENT_MODE"],
            L2_MODE=config_data["L2_MODE"],
            DROPOUT_MODE=config_data["DROPOUT_MODE"],
            OPTIMIZER=config_data["OPTIMIZER"],
            SCHEDULE_MODE=config_data["SCHEDULE_MODE"],
            EARLY_STOP_MODE=config_data["EARLY_STOP_MODE"],
            AVERAGE_MODE=config_data["AVERAGE_MODE"],
            TTA_MODE=config_data["TTA_MODE"],
            EPOCHS_COUNT=config_data["EPOCHS_COUNT"],
            BATCH_SIZE=config_data["BATCH_SIZE"]
        )


# Load default configuration from artifact/config/default.json
default_path = Path(__file__).parent / "artifact/config/default.json"
CONFIG = Config.load_config(default_path)


# Print module successfully executed
print("\n✅  config.py successfully executed")
