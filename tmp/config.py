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

    # Paths
    CHECKPOINT_PATH: Path
    CONFIG_PATH: Path
    DATA_PATH: Path
    ERROR_PATH: Path
    LOG_PATH: Path
    MODEL_PATH: Path
    RESULT_PATH: Path

    # Data and augmentation
    AUGMENT_MODE: dict

    # Regularization
    DROPOUT_MODE: dict
    L2_MODE: dict

    # Optimization
    OPTIMIZER: dict
    SCHEDULE_MODE: dict

    # Training logic
    AVERAGE_MODE: dict
    EARLY_STOP_MODE: dict
    TTA_MODE: dict

    # Core training params
    BATCH_SIZE: int
    EPOCHS_COUNT: int

    # Step 0: Load configuration from file
    @staticmethod
    def load_config(path: Path) -> "Config":
        """
        Function to load configuration from a JSON file.

        Args:
            path (Path): Full path to the JSON configuration file.

        Returns:
            Config: A fully initialized and validated Config object.
        """
        print("\n🎯  load_config is executing ...")                  # Step 0.1
        print("\n📂  Loading configuration file:")                 # Step 0.2
        print(f"→ {path}")

        # Step 0.3: Read JSON from file
        with open(path, "r") as f:
            config_data = json.load(f)

        # Step 0.4: Delegate to dictionary parser
        return Config._from_dict(config_data, path.parent)

    # Step 1: Construct Config from dictionary
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

        print("\n🎯  _from_dict is executing ...")

        # Step 1.1: Define required keys
        required_keys = [
            "CHECKPOINT_PATH", "CONFIG_PATH", "DATA_PATH", "ERROR_PATH",
            "LOG_PATH", "MODEL_PATH", "RESULT_PATH",
            "AUGMENT_MODE",
            "DROPOUT_MODE", "L2_MODE",
            "OPTIMIZER", "SCHEDULE_MODE",
            "AVERAGE_MODE", "EARLY_STOP_MODE", "TTA_MODE",
            "BATCH_SIZE", "EPOCHS_COUNT"
        ]

        # Step 1.2: Validate key presence
        missing = [key for key in required_keys if key not in config_data]
        if missing:
            raise ValueError(
                f"\n\n❌  ValueError from config.py at _from_dict()!\n"
                f"→ Missing keys in config: {missing}\n\n"
            )
        print("\n🛠️  Required configuration keys are present.")

        # Step 1.3: Resolve paths relative to the config module
        root_path = Path(__file__).parent

        # Step 1.4: Construct and return Config instance
        return Config(
            CHECKPOINT_PATH=root_path / config_data["CHECKPOINT_PATH"],
            CONFIG_PATH=root_path / config_data["CONFIG_PATH"],
            DATA_PATH=root_path / config_data["DATA_PATH"],
            ERROR_PATH=root_path / config_data["ERROR_PATH"],
            LOG_PATH=root_path / config_data["LOG_PATH"],
            MODEL_PATH=root_path / config_data["MODEL_PATH"],
            RESULT_PATH=root_path / config_data["RESULT_PATH"],

            AUGMENT_MODE=config_data["AUGMENT_MODE"],

            DROPOUT_MODE=config_data["DROPOUT_MODE"],
            L2_MODE=config_data["L2_MODE"],

            OPTIMIZER=config_data["OPTIMIZER"],
            SCHEDULE_MODE=config_data["SCHEDULE_MODE"],

            AVERAGE_MODE=config_data["AVERAGE_MODE"],
            EARLY_STOP_MODE=config_data["EARLY_STOP_MODE"],
            TTA_MODE=config_data["TTA_MODE"],

            BATCH_SIZE=config_data["BATCH_SIZE"],
            EPOCHS_COUNT=config_data["EPOCHS_COUNT"]
        )


# Step 2: Load default config on import
default_path = Path(__file__).parent / "artifact/config/default.json"
CONFIG = Config.load_config(default_path)

# Step 3: Confirm module execution
print("\n✅  config.py successfully executed.")
