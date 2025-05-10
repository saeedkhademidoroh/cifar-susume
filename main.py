# Import project-specific libraries
from experiment import run_experiment


# Run a single model (MobileNet) once
run_experiment(model_numbers=1, runs=1)

# Run Model 3 one time
# run_experiment(3)

# Run Models 3 to 5, each 5 times
# run_experiment((3, 5), runs=5)

# Run specific models 1, 3, and 5, each 2 times
# run_experiment([1, 3, 5], runs=2)

# Print confirmation message
print("\n✅ main.py successfully executed")