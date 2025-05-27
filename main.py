# Import project-specific modules
from experiment import run_pipeline
from utility import clean_old_output


# Print module execution banner
print("\n✅  main.py is being executed ...")

# Clean old outputs if CLEAN_MODE is enabled
clean_old_output(True)


# Define experiment pipeline: (model_number, config_name)
pipeline = [
    (9, "m9_base_res")
]

# Log pipeline execution summary
print("\n🚀  Launching experiment pipeline ...")
for model_number, config_name in pipeline:
    print(f"   • Model {model_number} with config '{config_name}'")

# Run experiments through pipeline
run_pipeline(pipeline)
