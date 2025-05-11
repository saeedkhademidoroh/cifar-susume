# Import standard libraries
import json
from pathlib import Path
from dataclasses import dataclass


# Dataclass to represent an immutable config loaded from config.json
@dataclass(frozen=True)
class Config:
    """
    Immutable configuration object for the CIFAR experiment system.

    All fields are populated from config.json at runtime. Paths are resolved
    relative to the location of this script.
    """

    CONFIG_PATH: Path
    DATA_PATH: Path
    LOG_PATH: Path
    CHECKPOINT_PATH: Path
    RESULT_PATH: Path
    EPOCHS_COUNT: int
    BATCH_SIZE: int
    CLEAN_LOG: bool
    CLEAN_CHECKPOINT: bool
    CLEAN_RESULT: bool
    LIGHT_MODE: bool

    @staticmethod
    def load_from_json() -> "Config":
        """
        Loads and validates configuration from config.json in the current directory.

        Returns:
            Config: A fully populated, immutable Config object.
        Raises:
            FileNotFoundError: If the config.json file does not exist.
            ValueError: If any required keys are missing in the config.
        """
        print("\n🎯 load_from_json")
        current_dir = Path(__file__).parent
        raw_config_path = current_dir / "config.json"

        if not raw_config_path.exists():
            raise FileNotFoundError(f"❌ FileNotFoundError:\nraw_config_path={raw_config_path}\n")

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
            "CLEAN_LOG",
            "CLEAN_CHECKPOINT",
            "CLEAN_RESULT",
            "LIGHT_MODE"
        ]

        missing = [key for key in required_keys if key not in config_data]
        if missing:
            raise ValueError(f"❌ ValueError:\nmissing={missing}\n")

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
            LIGHT_MODE=config_data.get("LIGHT_MODE")
        )


CONFIG = Config.load_from_json()

print("\n✅ config.py successfully executed")
