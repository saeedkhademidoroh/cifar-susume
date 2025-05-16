# Import standard libraries
import os
from experiment import run_pipeline

# Print confirmation message
print("\n✅ main.py is being executed")

# Disable GPU (force CPU usage)
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Define experiment pipeline: (model_number, config_name)
pipeline = [
    (0, "default"),
    (0, "desktop"),
    (1, "default"),
    (1, "desktop"),
    (2, "default"),
    (2, "desktop"),
    (3, "default"),
    (3, "desktop"),
    (4, "default"),
    (4, "desktop"),
]

# Run experiments through pipeline
run_pipeline(pipeline)
