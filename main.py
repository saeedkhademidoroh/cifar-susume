# Import project-specific modules
from data import load_dataset
from log import analyze_and_log_dataset
from config import load_config

config = load_config()
print(config["batch_size"])

# Load CIFAR-10 dataset into cifar/data/
train_loader, val_loader, test_loader = load_dataset()

# Analyze datasets and log to cifar/artifact/json/results.json
analyze_and_log_dataset(train_loader, "Train")
analyze_and_log_dataset(val_loader, "Validation")
analyze_and_log_dataset(test_loader, "Test")

# Print confirmation message
print("\n✅ main.py successfully executed")
