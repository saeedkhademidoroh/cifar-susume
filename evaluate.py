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


def _build_tta_transform(config):
    """
    Builds the transform pipeline for Test-Time Augmentation based on TTA_MODE.

    Returns:
        torchvision.transforms.Compose: TTA transform.
    """

    # Step 0: Print header for function execution
    print("\n🎯  build_tta_transform is executing ...")

    # Step 1: Extract TTA settings from config
    tta = config.TTA_MODE

    # Step 2: Log selected TTA augmentation policy
    print("\n🎛️  TTA augmentation policy:")
    print(f"→ random_crop enabled:  {tta.get('random_crop', False)}")
    print(f"→ random_flip enabled:  {tta.get('random_flip', False)}")
    print(f"→ cutout enabled:       {tta.get('cutout', False)}")  # Not applied here but logged for consistency

    # Step 3: Start transform list with PIL conversion
    ops = [transforms.ToPILImage()]

    # Step 4: Conditionally add augmentation operations
    if tta.get("random_crop", False):
        ops.append(transforms.RandomCrop(32, padding=4))  # Crop with padding
    if tta.get("random_flip", False):
        ops.append(transforms.RandomHorizontalFlip())     # Horizontal flip

    # Step 5: Append normalization steps (ToTensor + Normalize)
    ops += build_normalization_transform().transforms

    # Step 6: Return composed transform pipeline
    return transforms.Compose(ops)


# Function to evaluate trained model and extract metrics
def evaluate_model(model, history, test_data, test_labels, config, run_id, verbose=0):
    """
    Evaluates a trained model and extracts final metrics.

    Automatically recovers training history if not provided.
    Supports optional Test-Time Augmentation (TTA) if enabled in config.

    Args:
        model (tf.keras.Model): Trained model to evaluate.
        history (dict or object, optional): Training history or None to auto-load.
        test_data (np.ndarray): Test set inputs.
        test_labels (np.ndarray): Test set labels.
        config (SimpleNamespace): Configuration object with evaluation and logging settings.
        verbose (int): Verbosity for evaluation and prediction.

    Returns:
        dict: Dictionary with:
            - train_loss / train_acc
            - val_loss / val_acc (if available)
            - test_loss / test_acc
            - predictions
    """

    # Step 0: Print function header
    print("\n🎯  evaluate_model is executing ...")

    # Step 1: Verify test_data type
    if not isinstance(test_data, np.ndarray):
        raise ValueError(
            "\n\n❌  ValueError from evaluate.py at evaluate_model()!\n"
            "test_data must be a NumPy array\n\n"
        )

    # Step 2: Attempt fallback history load if missing
    if history is None and hasattr(model, "run_id"):
        history_path = config.CHECKPOINT_PATH / model.run_id / "history.json"
        if history_path.exists():
            try:
                with open(history_path, "r") as f:
                    history_data = json.load(f)
                history = SimpleNamespace(history=history_data)
                print(f"\n📄  Fallback training history loaded:\n→ Path: {history_path}")
            except Exception as e:
                print(f"\n\n❌  Failed to load fallback history:\n→ {e}\n\n")
                history = {}

    # Step 3: Try to extract training/validation metrics
    try:
        metrics = extract_history_metrics(history)
    except (ValueError, KeyError) as e:
        print(f"\n\n❌  Failed to extract history metrics:\n→ {e}\n\n")
        metrics = {
            "min_train_loss": None, "min_train_loss_epoch": None,
            "max_train_acc": None,  "max_train_acc_epoch": None,
            "min_val_loss": None,  "min_val_loss_epoch": None,
            "max_val_acc": None,   "max_val_acc_epoch": None,
        }

    # Step 4: Sanity check — validate model weights source
    best_weights_path = config.CHECKPOINT_PATH / run_id / "best.keras"
    if best_weights_path.exists():
        best_mtime = os.path.getmtime(best_weights_path)
        current_mtime = getattr(model, "_loaded_weights_mtime", None)
        if current_mtime and abs(current_mtime - best_mtime) > 1:
            print("\n\n❌  Model may not be loaded from best_model.h5\n")
        else:
            print("\n🔍  Model appears to originate from best checkpoint.")
    else:
        print("\n\n❌  No best_model.h5 found — unable to verify model source\n")

    # Step 5: Run baseline test evaluation
    final_test_loss, final_test_acc = model.evaluate(
        test_data,
        test_labels,
        batch_size=config.BATCH_SIZE,
        verbose=verbose
    )

    # Step 5.5: Print final test evaluation metrics
    print("\n📈  Baseline test evaluation complete")
    print(f"→ Final Test Loss:  {final_test_loss:.4f}")
    print(f"→ Final Test Acc:   {final_test_acc:.4f}")

    # Step 6: Apply Test-Time Augmentation if enabled
    if config.TTA_MODE.get("enabled", False):
        runs = config.TTA_MODE.get("runs", 5)
        print(
            f"\n🌀  Test-Time Augmentation is enabled"
            f"\n→ Running {runs} augmented passes per sample"
        )
        transform = _build_tta_transform(config)
        tta_preds = []

        for run_idx in range(runs):
            augmented = [transform((img * 255).astype(np.uint8)) for img in test_data]

            if run_idx == 0:
                print("\n🎛️  First test image is being TTA-transformed:")
                print(f"→ Original shape: {test_data[0].shape}")
                print(f"→ Original pixel range: min={test_data[0].min()}, max={test_data[0].max()}")
                arr = augmented[0].permute(1, 2, 0).numpy()
                print(f"→ Transformed shape: {arr.shape}")
                print(f"→ Transformed pixel range: min={arr.min():.3f}, max={arr.max():.3f}")

            batch = np.stack([img.permute(1, 2, 0).numpy() for img in augmented]).astype(np.float32)
            preds = model.predict(batch, verbose=verbose)
            tta_preds.append(preds)

        predictions = np.mean(tta_preds, axis=0)
        print(f"\n📈  TTA applied — predictions averaged over {runs} runs")

    else:
        # Step 7: Standard inference without TTA
        predictions = model.predict(test_data, verbose=verbose)

        # Step 7.5: Log prediction shape and dtype
        print(
            f"\n📊  Predictions collected (standard inference)"
            f"\n→ Shape: {predictions.shape} — DType: {predictions.dtype}"
        )

    # Step 8: Return full evaluation dictionary
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


# Print module successfully executed
print("\n✅  evaluate.py successfully executed.")
