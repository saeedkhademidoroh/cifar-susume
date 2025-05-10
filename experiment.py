# Standard imports
import datetime
from pathlib import Path

# Third-party imports
import json

# Project-specific imports
from data import load_dataset
from model import build_model
from train import train_model

# Create log directory
LOG_DIR = Path(__file__).parent / "artifact/log"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Generate timestamped log file
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
LOG_PATH = LOG_DIR / f"log_{timestamp}.txt"

# Open log file in append mode
log_file = open(LOG_PATH, "a")

def log_line(text=""):
    log_file.write(text + "\n")

# Function to store experiment result in JSON-compatible format
def collect_experiment_result(model, history, model_name: str):
    time = datetime.datetime.now().strftime("%H:%M:%S")
    layers_count = len(model.layers)
    optimizer = type(model.optimizer).__name__
    val_acc = max(history.history.get("val_accuracy", [0]))
    loss = min(history.history.get("loss", [0]))

    return {
        "model": model_name,
        "time": time,
        "layers": layers_count,
        "optimizer": optimizer,
        "val_accuracy": round(val_acc, 4),
        "min_loss": round(loss, 4)
    }

# Function to run experiments
def run_experiment(model_numbers, runs=1):
    train_data, train_labels, _, _ = load_dataset(one_hot=True)

    if isinstance(model_numbers, int):
        model_numbers = [model_numbers]
    elif isinstance(model_numbers, tuple):
        model_numbers = list(range(model_numbers[0], model_numbers[1] + 1))

    all_results = []

    for model_number in model_numbers:
        for run in range(1, runs + 1):
            model, description = build_model(model_number)
            trained_model, history = train_model(train_data, train_labels, model, model_name=f"m{model_number}")
            result = collect_experiment_result(trained_model, history, model_name=f"m{model_number}")
            all_results.append(result)

    # Write all results to JSON inside artifact/json
    json_dir = Path(__file__).parent / "artifact/json"
    json_dir.mkdir(parents=True, exist_ok=True)
    results_path = json_dir / f"results_{timestamp}.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)

    # Write log
    for r in all_results:
        log_line(json.dumps(r))

# Close log file after script finishes
log_file.close()

print("\n✅ experiment.py successfully executed")
