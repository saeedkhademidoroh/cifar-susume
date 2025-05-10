# Standard imports
import os
import sys
import json
import datetime
from pathlib import Path

# Third-party imports
import numpy as np

# Project-specific imports
from config import CONFIG
from data import load_dataset
from model import build_model
from train import train_model

# Function to collect result info
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

# Function to run experiment
def run_experiment(model_numbers, runs=1):
    """
    Executes training experiments and logs all stdout/stderr to timestamped file.
    """
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = CONFIG.LOG_PATH / f"log_{timestamp}.txt"
    CONFIG.LOG_PATH.mkdir(parents=True, exist_ok=True)

    # Redirect stdout/stderr globally
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    f = open(log_file, "a")
    sys.stdout = f
    sys.stderr = f

    try:
        train_data, train_labels, _, _ = load_dataset(one_hot=True)

        if isinstance(model_numbers, int):
            model_numbers = [model_numbers]
        elif isinstance(model_numbers, tuple):
            model_numbers = list(range(model_numbers[0], model_numbers[1] + 1))

        all_result = []

        # Ensure result directory exists and define shared result file path
        CONFIG.RESULT_PATH.mkdir(parents=True, exist_ok=True)
        result_file_path = CONFIG.RESULT_PATH / f"result_{timestamp}.json"

        for model_number in model_numbers:
            for run in range(1, runs + 1):
                model, description = build_model(model_number)

                trained_model, history = train_model(
                    train_data, train_labels,
                    model,
                    model_name=f"m{model_number}",
                    verbose=0,
                    result_file_path=result_file_path
                )

                result = collect_experiment_result(trained_model, history, model_name=f"m{model_number}")
                all_result.append(result)

        with open(result_file_path, "w") as jf:
            json.dump(all_result, jf, indent=2)


    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        f.close()

print("\n✅ experiment.py successfully executed\n")
