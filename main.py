# Import project-specific libraries
from experiment import run_experiment

# Run a single model (MobileNet) once
run_experiment(model_numbers=1, runs=1)

# -------------------- Test Case: experiment.run_experiment() --------------------

# Run Model 3 one time
# run_experiment(3)

# Run Models 3 to 5, each 5 times
# run_experiment((3, 5), runs=5)

# Run specific models 1, 3, and 5, each 2 times
# run_experiment([1, 3, 5], runs=2)

# -------------------- Test Case: data.log_to_json() ----------------------------

# train_data, train_labels, test_data, test_labels = load_dataset()

# log_to_json(key="train_overview", record={
#     "samples": len(train_data),
#     "shape": list(train_data[0].shape),
#     "labels": len(set(train_labels))
# })

# log_to_json(key="test_overview", record={
#     "samples": len(test_data),
#     "shape": list(test_data[0].shape),
#     "labels": len(set(test_labels))
# })

# print("\n✅ main.py successfully executed")

# -------------------- Test Case: config.CONFIG ---------------------------------

# print(f"\n🔧 Batch size from config: {CONFIG.BATCH_SIZE}")

# -------------------------------------------------------------------------------


# Print confirmation message
print("\n✅ main.py successfully executed")