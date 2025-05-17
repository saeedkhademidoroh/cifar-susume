# Import standard libraries
import os

# Import project-specific modules
from experiment import run_pipeline
from log import clean_old_outputs


# Print module execution banner
print("\n✅  main.py is being executed")

# Clean old outputs if CLEAN_MODE is enabled
clean_old_outputs(False)

# Force CPU usage by disabling GPU
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Define experiment pipeline: (model_number, config_name)
pipeline = [
    # (0, "default"),
    (0, "desktop"),
    # (0, "colab"),
    # (1, "default"),
    (1, "desktop"),
    # (1, "colab"),
    # (2, "default"),
    (2, "desktop"),
    # (2, "colab"),
    # (3, "default"),
    (3, "desktop"),
    # (3, "colab"),
    # (4, "default"),
    (4, "desktop"),
    # (4, "colab"),
    # (5, "default"),
    (5, "desktop"),
    # (5, "colab"),
]

# Run experiments through pipeline
run_pipeline(pipeline)
