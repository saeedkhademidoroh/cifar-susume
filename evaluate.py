# Third-party imports
import numpy as np
import matplotlib.pyplot as plt

# Project-specific imports
from config import CONFIG


# Function to evaluate model
def evaluate_model(model, history, test_data, test_labels, verbose=0):
    """
    Evaluates the model on test data, extracts training history, and displays key metrics.

    Parameters:
        model (tf.keras.Model): A trained model.
        history (tf.keras.callbacks.History or dict): Training history.
        test_data (numpy.ndarray): Feature set for test data.
        test_labels (numpy.ndarray): True labels for test data.
        verbose (int): Verbosity level for model evaluation (default: 1).

    Returns:
        dict: Contains min/max loss, accuracy, test loss, test accuracy, and predictions.
    """

    print("\n🎯 Evaluate Model 🎯")

    # Extract metrics from history
    metrics = extract_history_metrics(history)

    # Print training history details
    print("\n🔹 Training History:\n")
    print(f"Min Training Loss: {metrics['min_train_loss']:.4f} (Epoch {metrics['min_train_loss_epoch']})")
    print(f"Max Training Accuracy: {metrics['max_train_acc']:.4f} (Epoch {metrics['max_train_acc_epoch']})")

    if metrics["min_val_loss"] is not None:
        print(f"Min Validation Loss: {metrics['min_val_loss']:.4f} (Epoch {metrics['min_val_loss_epoch']})")
        print(f"Max Validation Accuracy: {metrics['max_val_acc']:.4f} (Epoch {metrics['max_val_acc_epoch']})")

    # Evaluate model
    final_test_loss, final_test_accuracy = model.evaluate(test_data, test_labels, batch_size=CONFIG.BATCH_SIZE, verbose=verbose)

    # Print evaluation results
    print("\n🔹 Evaluation Result:\n")
    print(f"Final Test Loss: {final_test_loss:.4f}")
    print(f"Final Test Accuracy: {final_test_accuracy:.4f}")

    # Predict values
    predictions = model.predict(test_data, verbose=verbose)

    # Return metrics for logging or further analysis
    return {
        "min_train_loss": metrics["min_train_loss"],
        "max_train_acc": metrics["max_train_acc"],
        "min_val_loss": metrics.get("min_val_loss"),
        "max_val_acc": metrics.get("max_val_acc"),
        "final_test_loss": final_test_loss,
        "final_test_accuracy": final_test_accuracy,
        "predictions": predictions,
    }


# Function to extract min/max loss & accuracy from history
def extract_history_metrics(history):
    """
    Extracts min/max loss and accuracy from training history.

    Parameters:
        history (tf.keras.callbacks.History or dict): The training history.

    Returns:
        dict: Contains min/max loss & accuracy with their corresponding epochs.
    """


    # Print header for function
    print("\n🎯 Extract History 🎯")


    # Ensure history is a dictionary
    history = history.history if hasattr(history, "history") else history

    # Extract min/max values and corresponding epochs
    metrics = {
        "min_train_loss": min(history["loss"]),
        "min_train_loss_epoch": history["loss"].index(min(history["loss"])) + 1,
        "max_train_acc": max(history["accuracy"]),
        "max_train_acc_epoch": history["accuracy"].index(max(history["accuracy"])) + 1,
    }

    # Check for validation data (optional)
    if "val_loss" in history and "val_accuracy" in history:
        metrics.update({
            "min_val_loss": min(history["val_loss"]),
            "min_val_loss_epoch": history["val_loss"].index(min(history["val_loss"])) + 1,
            "max_val_acc": max(history["val_accuracy"]),
            "max_val_acc_epoch": history["val_accuracy"].index(max(history["val_accuracy"])) + 1,
        })
    else:
        metrics["min_val_loss"], metrics["max_val_acc"] = None, None

    return metrics


# Print confirmation message
print("\n✅ evaluate.py successfully executed")