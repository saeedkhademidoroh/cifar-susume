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
    (3, "m3_base"),
    (3, "m3_drop"),
    (3, "m3_l2"),
    (3, "m3_reg"),
    (3, "m3_light"),
    (4, "m4_base"),
    (4, "m4_drop"),
    (4, "m4_l2"),
    (4, "m4_reg"),
    (4, "m4_light")
]


# Run experiments through pipeline
run_pipeline(pipeline)
