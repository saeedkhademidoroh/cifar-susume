# Import standard libraries
import json
import os
from types import SimpleNamespace

# Import third-party libraries
import numpy as np
from torchvision import transforms

# Import project-specific libraries
from data import build_normalization_transform
from utility import extract_history_metrics

# Function to build tta transform
def _build_tta_transform(config):
    """
    Builds the transform pipeline for Test-Time Augmentation based on TTA_MODE.

    Returns:
        torchvision.transforms.Compose: TTA transform.
    """

    # Print header for function execution
    print("\n🎯  build_tta_transform is executing ...")

    # Extract TTA settings from config
    tta = config.TTA_MODE

    # Log TTA augmentation policy
    print("\n🎛️  TTA augmentation policy:")
    print(f"→ random_crop enabled:  {tta.get('random_crop', False)}")
    print(f"→ random_flip enabled:  {tta.get('random_flip', False)}")
    print(f"→ cutout enabled:       {tta.get('cutout', False)}")

    # Initialize transform ops with image format conversion
    ops = [transforms.ToPILImage()]

    # Conditionally add augmentations
    if tta.get("random_crop", False):
        ops.append(transforms.RandomCrop(32, padding=4))  # Safe spatial jitter
    if tta.get("random_flip", False):
        ops.append(transforms.RandomHorizontalFlip())     # Safe horizontal flip

    # Append normalization steps (ToTensor + Normalize)
    ops += build_normalization_transform().transforms

    # Return full transform pipeline
    return transforms.Compose(ops)


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
    print("\n🎯  evaluate_model is executing ...")

    # Sanity check — ensure test input is in NumPy format
    if not isinstance(test_data, np.ndarray):
        raise ValueError("\n\n❌  ValueError from evaluate.py at evaluate_model()!\ntest_data must be a NumPy array\n\n")


    # Load fallback history if history is missing and model.run_id is available
    if history is None and hasattr(model, "run_id"):
        history_path = config.CHECKPOINT_PATH / model.run_id / "history.json"

        if history_path.exists():
            try:
                # Load the JSON history file into a SimpleNamespace
                with open(history_path, "r") as f:
                    history_data = json.load(f)

                history = SimpleNamespace(history=history_data)

                # Print fallback path info
                print(f"\n📄  Fallback training history loaded:")
                print(f"→ Path: {history_path}")
            except Exception as e:
                # Print fallback load failure and continue with empty dict
                print(f"\n\n❌  Error from evaluate.py at evaluate_model()!\nFailed to load fallback history:\n→ {e}\n\n")
                history = {}

    # Extract metrics from training history (min/max train/val loss/acc)
    try:
        metrics = extract_history_metrics(history)

    except (ValueError, KeyError) as e:
        # If metrics can't be extracted, fall back to null values
        print(f"\n\n❌  Error from evaluate.py at evaluate_model()!\nFailed to extract history metrics:\n→ {e}\n\n")
        metrics = {
            "min_train_loss": None,
            "min_train_loss_epoch": None,
            "max_train_acc": None,
            "max_train_acc_epoch": None,
            "min_val_loss": None,
            "min_val_loss_epoch": None,
            "max_val_acc": None,
            "max_val_acc_epoch": None,
        }

    # Soft check: confirm model source is best checkpoint (non-intrusive)
    best_weights_path = config.CHECKPOINT_PATH / model.run_id / "best_model.h5"

    if best_weights_path.exists():
        best_mtime = os.path.getmtime(best_weights_path)
        current_mtime = getattr(model, "_loaded_weights_mtime", None)

        if current_mtime and abs(current_mtime - best_mtime) > 1:
            print(f"\n\n❌  Error from evaluate.py at evaluate_model()!\nModel may not be loaded from best_model.h5\n\n")
        else:
            print(f"\n🔍  Model appears to originate from best checkpoint.")
    else:
        print(f"\n\n❌  Error from evaluate.py at evaluate_model()!\nNo best_model.h5 found — unable to verify model source\n\n")

    # Evaluate final test performance (loss and accuracy)
    final_test_loss, final_test_acc = model.evaluate(
        test_data,
        test_labels,
        batch_size=config.BATCH_SIZE,
        verbose=verbose
    )

    # If TTA is enabled, run multiple augmented inference passes
    if config.TTA_MODE.get("enabled", False):
        runs = config.TTA_MODE.get("runs", 5)  # Number of TTA passes

        # Load transform pipeline with augmentation and normalization
        transform = _build_tta_transform(config)
        tta_preds = []

        for run_idx in range(runs):
            # Transform each test image individually (back to uint8 first)
            augmented = [
                transform((img * 255).astype(np.uint8)) for img in test_data
            ]

            # 🔍 Log verification info only for the first TTA run and first image
            if run_idx == 0:
                print(f"\n🎛️  First test image is being TTA-transformed:")
                print(f"→ Original shape: {test_data[0].shape}")
                print(f"→ Original pixel range: min={test_data[0].min()}, max={test_data[0].max()}")

                arr = augmented[0].permute(1, 2, 0).numpy()
                print(f"→ Transformed shape: {arr.shape}")
                print(f"→ Transformed pixel range: min={arr.min():.3f}, max={arr.max():.3f}")

            # Convert each image back to NHWC float32 format
            batch = np.stack([img.permute(1, 2, 0).numpy() for img in augmented]).astype(np.float32)

            # Predict on the augmented test batch
            preds = model.predict(batch, verbose=verbose)
            tta_preds.append(preds)

        # Average predictions from all TTA runs
        predictions = np.mean(tta_preds, axis=0)
        print(f"\n📈  TTA applied — predictions averaged over {runs} runs")

    else:
        # Standard inference without TTA
        predictions = model.predict(test_data, verbose=verbose)

    # Verbose prediction shape (optional)
    print(
        f"\n📊  Predictions Summary"
        f"\n→ TTA enabled     : {config.TTA_MODE.get('enabled', False)}"
        f"\n→ Predictions shape: {predictions.shape}\n"
    )

    # Return structured output including metrics, performance, and predictions
    return {
        # Training stats
        "min_train_loss": metrics["min_train_loss"],
        "min_train_loss_epoch": metrics["min_train_loss_epoch"],
        "max_train_acc": metrics["max_train_acc"],
        "max_train_acc_epoch": metrics["max_train_acc_epoch"],

        # Validation stats (if any)
        "min_val_loss": metrics.get("min_val_loss"),
        "min_val_loss_epoch": metrics.get("min_val_loss_epoch"),
        "max_val_acc": metrics.get("max_val_acc"),
        "max_val_acc_epoch": metrics.get("max_val_acc_epoch"),

        # Final test scores
        "final_test_loss": final_test_loss,
        "final_test_acc": final_test_acc,

        # Prediction matrix (used for TTA voting or further processing)
        "predictions": predictions,
    }


# Print module successfully executed
print("\n✅  evaluate.py successfully executed")
