# Import standard libraries
import os

# Import project-specific modules
from experiment import run_pipeline
from log import clean_old_output


# Print module execution banner
print("\n✅  main.py is being executed")

# Clean old outputs if CLEAN_MODE is enabled
clean_old_output(False)

# Force CPU usage by disabling GPU
# os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Define experiment pipeline: (model_number, config_name)
pipeline = [
    (9, "m9_base"),
    (6, "m6_legacy"),
    (6, "m6_rebase"),
    (9, "m9_tuned"),
    (9, "m9_drop")
    # (9, "default"),
    # (6, "default"),
    # (6, "default"),
    # (9, "default"),
    # (9, "default")
]


# Run experiments through pipeline
run_pipeline(pipeline)
