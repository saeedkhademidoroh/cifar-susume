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
    (1, "m1_base"),
    (1, "m1_l2"),
    (1, "m1_drop"),
    (1, "m1_light"),
    (1, "m1_reg"),
]

# Run experiments through pipeline
run_pipeline(pipeline)
