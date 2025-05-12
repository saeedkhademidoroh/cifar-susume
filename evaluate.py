# Import standard libraries
import numpy as np
import matplotlib.pyplot as plt

# Import project-specific libraries
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

    # Print header for function execution
    print("\n🎯 evaluate_model")

    # Extract metrics from history
    metrics = extract_history_metrics(history)

    # Evaluate model
    final_test_loss, final_test_accuracy = model.evaluate(test_data, test_labels, batch_size=CONFIG.BATCH_SIZE, verbose=verbose)

    # Predict values
    predictions = model.predict(test_data, verbose=verbose)

    # Print summary of training and evaluation metrics
    print("\n📊 Summary Metrics:")
    print(f"Min Training Loss       : {metrics['min_train_loss']:.4f} (Epoch {metrics['min_train_loss_epoch']})")
    print(f"Max Training Accuracy   : {metrics['max_train_acc']:.4f} (Epoch {metrics['max_train_acc_epoch']})")
    if metrics["min_val_loss"] is not None:
        print(f"Min Validation Loss     : {metrics['min_val_loss']:.4f} (Epoch {metrics['min_val_loss_epoch']})")
        print(f"Max Validation Accuracy : {metrics['max_val_acc']:.4f} (Epoch {metrics['max_val_acc_epoch']})")
    print(f"Final Test Loss         : {final_test_loss:.4f}")
    print(f"Final Test Accuracy     : {final_test_accuracy:.4f}")

    # Return metrics for logging and further analysis
    return {
        "min_train_loss": metrics["min_train_loss"],
        "max_train_acc": metrics["max_train_acc"],
        "min_val_loss": metrics.get("min_val_loss"),
        "max_val_acc": metrics.get("max_val_acc"),
        "final_test_loss": final_test_loss,
        "final_test_accuracy": final_test_accuracy,
        "predictions": predictions,
    }


# Function to extract history metrics
def extract_history_metrics(history):
    """
    Extracts min/max loss and accuracy from training history.

    Parameters:
        history (tf.keras.callbacks.History or dict): The training history.

    Returns:
        dict: Contains min/max loss & accuracy with their corresponding epochs.
    """

    # Print header for function execution
    print("\n🎯 extract_history_metrics\n")

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

    # Return metrics dictionary
    return metrics


# Print confirmation message
print("\n✅ evaluate.py successfully executed\n")