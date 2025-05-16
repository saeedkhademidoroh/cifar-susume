# Import standard libraries
import json


# Function to evaluate model
def evaluate_model(model, history, test_data, test_labels, config, verbose=0):
    """
    Evaluates the model on test data, extracts training history, and displays key metrics.

    Parameters:
        model (tf.keras.Model): A trained model.
        history (tf.keras.callbacks.History or dict): Training history.
        test_data (numpy.ndarray): Feature set for test data.
        test_labels (numpy.ndarray): True labels for test data.
        config (Config): Configuration object containing BATCH_SIZE and CHECKPOINT_PATH.
        verbose (int): Verbosity level for model evaluation (default: 1).

    Returns:
        dict: Contains min/max loss, accuracy, test loss, test accuracy, and predictions.
    """

    # Print header for function execution
    print("\n🎯 evaluate_model")

    # Fallback: Load saved history if not in memory (resumed run)
    if history is None:
        history_path = config.CHECKPOINT_PATH / f"m{model.model_id}/history.json"
        if history_path.exists():
            try:
                with open(history_path, "r") as f:
                    history_data = json.load(f)
                    class DummyHistory: pass
                    history = DummyHistory()
                    history.history = history_data
                print(f"\n📄 Loaded fallback history from:\n{history_path}")
            except Exception as e:
                print(f"\n⚠️ Failed to load fallback history:\n{e}")
                history = {}

    # Extract metrics from history
    metrics = extract_history_metrics(history)

    # Evaluate model
    final_test_loss, final_test_accuracy = model.evaluate(test_data, test_labels, batch_size=config.BATCH_SIZE, verbose=verbose)

    # Predict values
    predictions = model.predict(test_data, verbose=verbose)

    # Build result dictionary for both JSON and log output
    return {
        "min_train_loss": metrics["min_train_loss"],
        "min_train_loss_epoch": metrics["min_train_loss_epoch"],
        "max_train_acc": metrics["max_train_acc"],
        "max_train_acc_epoch": metrics["max_train_acc_epoch"],
        "min_val_loss": metrics.get("min_val_loss"),
        "min_val_loss_epoch": metrics.get("min_val_loss_epoch"),
        "max_val_acc": metrics.get("max_val_acc"),
        "max_val_acc_epoch": metrics.get("max_val_acc_epoch"),
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
    print("\n🎯 extract_history_metrics")

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
