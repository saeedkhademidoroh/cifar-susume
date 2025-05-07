# Import standard libraries
import os
import time
import json
import platform
from datetime import datetime

# Import third-party libraries
import numpy as np
import torch

# Import project-specific modules
from config import LOG_PATH

# Ensure the output directory exists
def ensure_dir(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)

# Function to log a dictionary under a named key in JSON
def log_to_json(file_path, key, record):
    ensure_dir(file_path)

    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            data = json.load(f)
    else:
        data = {}

    if "timestamp" not in record:
        record["timestamp"] = time.time()
        record["timestamp_utc"] = datetime.utcfromtimestamp(record["timestamp"]).isoformat()

    if key not in data:
        data[key] = []

    data[key].append(record)

    with open(file_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"✅ Logged under '{key}' in {file_path}")

# Function to log dataset statistics
def log_dataset_stats(all_pixels, name, json_path, sample_count):
    record = {
        "shape": list(all_pixels.shape),
        "dtype": str(all_pixels.dtype),
        "missing_values": int(np.isnan(all_pixels).sum()),
        "min": float(np.min(all_pixels)),
        "max": float(np.max(all_pixels)),
        "mean": float(np.mean(all_pixels)),
        "std": float(np.std(all_pixels)),
        "sample_count": sample_count,
        "normalization": "mean=0.5, std=0.5",
        "torch_version": torch.__version__,
        "platform": platform.node()
    }

    log_to_json(json_path, f"{name.lower()}_stats", record)

# Function to analyze and optionally log a dataset
def analyze_and_log_dataset(loader, name="Dataset", json_path=LOG_PATH, verbose=True):
    """
    Analyzes a DataLoader (flattened pixel stats) and logs them.

    Args:
        loader (DataLoader): PyTorch DataLoader to analyze.
        name (str): Dataset name.
        json_path (Path): Output JSON file path.
        verbose (bool): Whether to print stats to console.
    """
    pixel_values = []
    for images, _ in loader:
        images = images.view(images.size(0), -1)
        pixel_values.append(images.numpy())

    all_pixels = np.concatenate(pixel_values, axis=0)

    if verbose:
        print(f"\n🎯 {name} Analysis 🎯")
        print("\n🔹 Shape & Dtype:")
        print(f"Images: {all_pixels.shape}, Dtype: {all_pixels.dtype}")
        print("\n🔹 Missing Values:")
        print(f"NaNs: {np.isnan(all_pixels).sum()}")
        print("\n🔹 Pixel Statistics:")
        print(f"Min: {np.min(all_pixels):.4f}")
        print(f"Max: {np.max(all_pixels):.4f}")
        print(f"Mean: {np.mean(all_pixels):.4f}")
        print(f"Std: {np.std(all_pixels):.4f}")

    log_dataset_stats(all_pixels, name, json_path, sample_count=all_pixels.shape[0])
