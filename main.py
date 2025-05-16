# Import standard libraries
import os
from experiment import run_pipeline
from log import clean_old_outputs

# Print confirmation message
print("\n✅ main.py is being executed")

# Clean old outputs
clean_old_outputs(True)

# Disable GPU (force CPU usage)
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Define experiment pipeline: (model_number, config_name)
pipeline = [
    (0, "default"),
    (1, "default"),
    (2, "default"),
    (3, "default"),
    (4, "default"),
]

# Run experiments through pipeline
run_pipeline(pipeline)