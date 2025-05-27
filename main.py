# Import standard libraries
# import os

# Import project-specific modules
from experiment import run_pipeline
from log import clean_old_output


# Print module execution banner
print("\n✅  main.py is being executed")

# Clean old outputs if CLEAN_MODE is enabled
clean_old_output(True)

# Force CPU usage by disabling GPU
# os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Define experiment pipeline: (model_number, config_name)
pipeline = [
    (9, "m9_base_res")
]

# Run experiments through pipeline
run_pipeline(pipeline)