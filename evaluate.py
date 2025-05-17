# Import standard libraries
import json


# Function to evaluate trained model
def evaluate_model(model, history, test_data, test_labels, config, verbose=0):
    """
    Function to evaluate a trained model and extract relevant training/test metrics.

    Loads fallback history if missing, evaluates the model on test data,
    and prepares key statistics for logging and result tracking.

    Args:
        model (tf.keras.Model): Trained model instance
        history (History or dict or None): Training history or None to trigger fallback loading
        test_data (np.ndarray): Test dataset features
        test_labels (np.ndarray): Test dataset labels
        config (Config): Config object with BATCH_SIZE and CHECKPOINT_PATH
        verbose (int): Verbosity level for evaluation (default: 0)

    Returns:
        dict: Dictionary containing training stats, test performance, and predictions
    """

    # Print header for function execution
    print("\n🎯  evaluate_model")

    # Fallback: attempt to load history from saved JSON if not provided
    if history is None:
        history_path = config.CHECKPOINT_PATH / f"m{model.model_id}/history.json"
        if history_path.exists():
            try:
                with open(history_path, "r") as f:
                    history_data = json.load(f)

                class DummyHistory:
                    pass

                history = DummyHistory()
                history.history = history_data
                print(f"\n📄 Loading fallback history:\n{history_path}")
            except Exception as e:
                print(f"\n⚠️ Failing to load fallback history:\n{e}")
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

    # Predict outputs on test data
    predictions = model.predict(test_data, verbose=verbose)

    # Package all metrics into a dictionary
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
        "final_test_acc": final_test_acc,
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
        # Fallback when validation data is not available
        metrics["min_val_loss"], metrics["max_val_acc"] = None, None

    return metrics  # Return dictionary of extracted metrics


# Print module successfully executed
print("\n✅  evaluate.py successfully executed")
