# Import standard libraries
# import os

# Import project-specific modules
from experiment import run_pipeline
from log import clean_old_output


# Print module execution banner
print("\n✅  main.py is being executed")

# Clean old outputs if CLEAN_MODE is enabled
# clean_old_output(False)

# Force CPU usage by disabling GPU
# os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Define experiment pipeline: (model_number, config_name)
pipeline = [
    (6, "m6_legacy"),         # Legacy config (nonstandard but informative)
    (6, "m6_rebase_res"),     # ResNet-faithful baseline for model 6
    (6, "m6_rebase_mod"),     # Modernized model 6
    (9, "m9_base_res"),       # ResNet-faithful baseline
    (9, "m9_base_mod")        # Modernized model 9
]

# Run experiments through pipeline
run_pipeline(pipeline)
