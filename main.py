# Import project-specific modules
from experiment import run_pipeline
from utility import clean_old_output

# Step 0: Print module execution banner
print("\n✅  main.py is being executed ...")

# Step 1: Clean old output directories if cleanup is enabled
clean_old_output(True)

# Step 2: Define experiment pipeline (each item = (model_number, config_name))
pipeline = [
    (9, "default")
]

# Step 3: Log pipeline summary before execution
print("\n🚀  Launching experiment pipeline ...")
for model_number, config_name in pipeline:
    print(f"→ Model {model_number} with config '{config_name}'")

# Step 4: Execute the defined experiment pipeline
run_pipeline(pipeline)
