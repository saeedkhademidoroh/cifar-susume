# Import standard libraries
import json
from types import SimpleNamespace

# Import third-party libraries
import numpy as np
from PIL import Image

# Import project-specific libraries
from data import build_augmentation_transform


# Function to evaluate trained model
def evaluate_model(model, history, test_data, test_labels, config, verbose=0):
    """
    Function to evaluate a trained model and extract training/test performance.

    Handles optional TTA (Test-Time Augmentation), loads fallback training history
    if not provided, and computes performance metrics.

    Args:
        model (tf.keras.Model): Trained model instance.
        history (History or dict or None): Training history, or None to auto-load from disk.
        test_data (np.ndarray): Test images.
        test_labels (np.ndarray): Test labels.
        config (Config): Configuration object with paths and evaluation settings.
        verbose (int): Verbosity level during evaluation and prediction.

    Returns:
        dict: Dictionary of training stats, final test loss/accuracy, and predictions.
    """

    # Print header for function execution
    print("\n🎯  evaluate_model")

    # Ensure test_data is a NumPy array
    assert isinstance(test_data, np.ndarray), "test_data must be a NumPy array"

    # Attempt to load fallback history if missing
    if history is None and hasattr(model, "run_id"):
        history_path = config.CHECKPOINT_PATH / model.run_id / "history.json"

        if history_path.exists():
            try:
                with open(history_path, "r") as f:
                    history_data = json.load(f)

                history = SimpleNamespace(history=history_data)
                print(f"\n📄 Loading fallback history:\n{history_path}")
            except Exception as e:
                print(f"\n⚠️  Failing to load fallback history:\n{e}")
                history = {}

    # Extract metrics from training history
    metrics = extract_history_metrics(history)


    # Evaluate model on test data
    final_test_loss, final_test_acc = model.evaluate(
        test_data,
        test_labels,
        batch_size=config.BATCH_SIZE,
        verbose=verbose
    )

    # Predict outputs using Test-Time Augmentation if enabled
    if config.TTA_MODE.get("enabled", False):
        # Number of augmentation passes per test sample
        runs = config.TTA_MODE.get("runs", 5)

        # Build transform pipeline with augmentation + normalization
        transform = build_augmentation_transform(config)

        tta_preds = []

        for _ in range(runs):
            # Apply transform to each test sample (after converting to uint8)
            augmented = [
                transform((img * 255).astype(np.uint8)) for img in test_data
            ]

            # Convert transformed tensors back to NumPy arrays in NHWC format
            batch = np.stack([img.permute(1, 2, 0).numpy() for img in augmented]).astype(np.float32)

            # Predict on the augmented batch
            preds = model.predict(batch, verbose=verbose)
            tta_preds.append(preds)

        # Average predictions across all augmentation passes
        predictions = np.mean(tta_preds, axis=0)
        print(f"\n📈  TTA applied — averaged over {runs} runs")
    else:
        # Standard prediction without augmentation
        predictions = model.predict(test_data, verbose=verbose)

    # Package training metrics, test performance, and predictions
    return {
        # Training metrics
        "min_train_loss": metrics["min_train_loss"],
        "min_train_loss_epoch": metrics["min_train_loss_epoch"],
        "max_train_acc": metrics["max_train_acc"],
        "max_train_acc_epoch": metrics["max_train_acc_epoch"],

        # Validation metrics (if available)
        "min_val_loss": metrics.get("min_val_loss"),
        "min_val_loss_epoch": metrics.get("min_val_loss_epoch"),
        "max_val_acc": metrics.get("max_val_acc"),
        "max_val_acc_epoch": metrics.get("max_val_acc_epoch"),

        # Final test performance
        "final_test_loss": final_test_loss,
        "final_test_acc": final_test_acc,

        # Raw predictions
        "predictions": predictions,
    }


# Function to extract history metrics
def extract_history_metrics(history):
    """
    Function to extract min/max training and validation metrics from history.

    Handles both `History` objects and plain dictionaries. Computes:
    - Minimum training loss and corresponding epoch
    - Maximum training accuracy and corresponding epoch
    - (Optional) Minimum validation loss and maximum validation accuracy with epochs

    Args:
        history (History or dict): Training history object or dictionary

    Returns:
        dict: Dictionary containing key metrics and their epochs
    """

    # Print header for function execution
    print("\n🎯  extract_history_metrics")

    # Convert to raw dict if History object is provided
    history = history.history if hasattr(history, "history") else history

    # Extract training metrics
    metrics = {
        "min_train_loss": min(history["loss"]),
        "min_train_loss_epoch": history["loss"].index(min(history["loss"])) + 1,
        "max_train_acc": max(history["accuracy"]),
        "max_train_acc_epoch": history["accuracy"].index(max(history["accuracy"])) + 1,
    }

    # Extract validation metrics if available
    if "val_loss" in history and "val_accuracy" in history:
        metrics.update({
            "min_val_loss": min(history["val_loss"]),
            "min_val_loss_epoch": history["val_loss"].index(min(history["val_loss"])) + 1,
            "max_val_acc": max(history["val_accuracy"]),
            "max_val_acc_epoch": history["val_accuracy"].index(max(history["val_accuracy"])) + 1,
        })
    else:
        metrics["min_val_loss"] = None
        metrics["min_val_loss_epoch"] = None
        metrics["max_val_acc"] = None
        metrics["max_val_acc_epoch"] = None


    return metrics  # Return dictionary of extracted metrics


# Print module successfully executed
print("\n✅  evaluate.py successfully executed")
