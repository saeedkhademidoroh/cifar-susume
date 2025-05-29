# Import standard libraries
import datetime
import json
import os
from pathlib import Path
import sys
import time
import traceback

# Import third-party libraries
from keras.api.models import load_model, Model
import numpy as np

# Import project-specific modules
from config import CONFIG
from data import build_dataset
from evaluate import evaluate_model, extract_history_metrics
from model import build_model
from train import train_model
from utility import (
    ensure_output_directories,
    initialize_logging,
    load_previous_results,
    log_to_json,
    to_json_compatible,
    recover_training_history,
)


def _create_evaluation_dictionary(model_number, run, config_name, duration, config, metrics, test_loss, test_accuracy):
    """
    Constructs a structured dictionary capturing experiment metadata and performance.

    Includes: model details, full hyperparameters, metric summaries, and test results.

    Args:
        model_number (int): Index of the model variant used.
        run (int): Run ID for this particular execution.
        config_name (str): Name of the config file used.
        duration (float): Total runtime in seconds.
        config (Config): Loaded config object with hyperparameters.
        metrics (dict): Dictionary of training/validation metrics.
        test_loss (float): Final test loss value.
        test_accuracy (float): Final test accuracy value.

    Returns:
        dict: Complete experiment result entry to be logged.
    """

    # Step 0: Print function execution header
    print("\n🎯  _create_evaluation_dictionary is executing ...")

    # Step 1: Get current Tehran time directly
    tehran_tz = datetime.timezone(datetime.timedelta(hours=3, minutes=30))
    now_tehran = datetime.datetime.now(tehran_tz)

    # Step 2: Build and return the structured experiment result dictionary
    return {

        # Step 3: Metadata about the experiment run
        "model": model_number,
        "run": run,
        "config_name": config_name,
        "date": now_tehran.strftime("%Y-%m-%d"),  # Tehran date
        "time": now_tehran.strftime("%H:%M:%S"),  # Tehran time
        "duration": str(datetime.timedelta(seconds=int(duration))),  # Elapsed time

        # Step 4: Explicitly unpacked configuration fields
        "parameters": {
            "LIGHT_MODE": getattr(config, "LIGHT_MODE", False),
            "FROZEN_BN": getattr(config, "FROZEN_BN", False),

            "AUGMENT_MODE": {
                "enabled": config.AUGMENT_MODE.get("enabled", False),
                "random_crop": config.AUGMENT_MODE.get("random_crop", False),
                "random_flip": config.AUGMENT_MODE.get("random_flip", False),
                "cutout": config.AUGMENT_MODE.get("cutout", False)
            },

            "DROPOUT_MODE": {
                "enabled": config.DROPOUT_MODE.get("enabled", False),
                "rate": config.DROPOUT_MODE.get("rate", 0.0)
            },

            "L2_MODE": {
                "enabled": config.L2_MODE.get("enabled", False),
                "lambda": config.L2_MODE.get("lambda", 0.0),
                "mode": config.L2_MODE.get("mode", None)
            },

            "OPTIMIZER": {
                "type": config.OPTIMIZER.get("type", "sgd"),
                "learning_rate": config.OPTIMIZER.get("learning_rate", 0.01),
                "momentum": config.OPTIMIZER.get("momentum", 0.0)
            },

            "SCHEDULE_MODE": {
                "enabled": config.SCHEDULE_MODE.get("enabled", False),
                "type": config.SCHEDULE_MODE.get("type", None),
                "gamma": config.SCHEDULE_MODE.get("gamma", None),
                "milestones": config.SCHEDULE_MODE.get("milestones", None),
                "verbose": config.SCHEDULE_MODE.get("verbose", None),
                "warmup_epochs": config.SCHEDULE_MODE.get("warmup_epochs", None),
                "factor": config.SCHEDULE_MODE.get("factor", None),
                "patience": config.SCHEDULE_MODE.get("patience", None),
                "min_lr": config.SCHEDULE_MODE.get("min_lr", None)
            },

            "EARLY_STOP_MODE": {
                "enabled": config.EARLY_STOP_MODE.get("enabled", False),
                "monitor": config.EARLY_STOP_MODE.get("monitor", None),
                "patience": config.EARLY_STOP_MODE.get("patience", None),
                "restore_best_weights": config.EARLY_STOP_MODE.get("restore_best_weights", None),
                "verbose": config.EARLY_STOP_MODE.get("verbose", None)
            },

            "AVERAGE_MODE": {
                "enabled": config.AVERAGE_MODE.get("enabled", False),
                "start_epoch": config.AVERAGE_MODE.get("start_epoch", None)
            },

            "TTA_MODE": {
                "enabled": config.TTA_MODE.get("enabled", False),
                "random_crop": config.TTA_MODE.get("random_crop", False),
                "random_flip": config.TTA_MODE.get("random_flip", False),
                "runs": config.TTA_MODE.get("runs", 1)
            },

            "EPOCHS_COUNT": getattr(config, "EPOCHS_COUNT", None),
            "BATCH_SIZE": getattr(config, "BATCH_SIZE", None)

            # Future option for reproducibility:
            # "SEED": getattr(config, "SEED", None)
        },

        # Step 5: Key training and evaluation metrics
        "min_train_loss": metrics.get("min_train_loss", None),
        "min_train_loss_epoch": metrics.get("min_train_loss_epoch", None),
        "max_train_acc": metrics.get("max_train_acc", None),
        "max_train_acc_epoch": metrics.get("max_train_acc_epoch", None),
        "min_val_loss": metrics.get("min_val_loss", None),
        "min_val_loss_epoch": metrics.get("min_val_loss_epoch", None),
        "max_val_acc": metrics.get("max_val_acc", None),
        "max_val_acc_epoch": metrics.get("max_val_acc_epoch", None),
        "final_test_loss": test_loss,
        "final_test_acc": test_accuracy
    }


def _run_single_pipeline_entry(model_number, config_path, config_name, run, timestamp, completed_triplets, all_results, result_file):
    """
    Executes one complete experiment loop:
    - Loads configuration and dataset
    - Builds model and trains (or resumes)
    - Evaluates and logs results
    """

    # Step 0: Print function execution header
    print("\n🎯  _run_single_pipeline_entry is executing ...")

    # Step 1: Load configuration file
    config = CONFIG.load_config(config_path)

    # Step 2: Verify config is of correct type
    if not isinstance(config, CONFIG.__class__):
        raise TypeError(
            f"\n\n❌  Error from experiment.py at _run_single_pipeline_entry()!\n"
            f"→ Config loading failed: expected instance of CONFIG class\n→ Got: {type(config)}\n\n"
        )

    # Step 3: Ensure all output directories exist
    ensure_output_directories(config)

    # Step 4: Skip if experiment already completed
    if (model_number, run, config_name) in completed_triplets:
        print(f"\n⏩  Skipping experiment m{model_number}_r{run} with '{config_name}' (already completed)")
        return result_file

    # Step 5: Announce experiment start
    print(f"\n🚀  Launching experiment m{model_number}_r{run} with '{config_name}'")
    start_time = time.time()
    run_id = f"m{model_number}_r{run}_{config_name}"

    try:
        # Step 6: Load training and test dataset
        train_data, train_labels, val_data, val_labels, test_data, test_labels = build_dataset(config)


        # Step 6b: Verify all components are numpy arrays
        for var_name, var in zip(
            ["train_data", "train_labels", "test_data", "test_labels"],
            [train_data, train_labels, test_data, test_labels]
        ):
            if not isinstance(var, np.ndarray):
                raise TypeError(
                    f"\n\n❌  Error from experiment.py at _run_single_pipeline_entry()!\n"
                    f"→ Dataset component '{var_name}' must be a numpy array\n→ Got: {type(var)}\n\n"
                )

        # Step 7: Build model for current variant
        model = build_model(model_number, config)
        if not isinstance(model, Model):
            raise TypeError(
                f"\n\n❌  Error from experiment.py at _run_single_pipeline_entry()!\n"
                f"→ build_model() did not return a tf.keras.Model instance\n→ Got: {type(model)}\n\n"
            )

        # Step 8: Train or resume the model
        print("\n🏋️  Training model (or resuming if applicable) ...")
        trained_model, history, resumed = train_model(
            train_data, train_labels, model, model_number, run, config_name, config,
            val_data=val_data, val_labels=val_labels
        )
        if not isinstance(resumed, bool):
            raise TypeError(
                f"\n\n❌  Error from experiment.py at _run_single_pipeline_entry()!\n"
                f"→ train_model() returned invalid 'resumed' flag\n→ Got: {type(resumed)}\n\n"
            )

        # Step 9: Log resume status and recover training history if needed
        if resumed:
            print("\n🔁  Resume detected — checking for missing history ...")
        if resumed and (history is None or not hasattr(history, "history")):
            print("\n📂  In-memory history is missing — attempting to recover from checkpoint")
            history = recover_training_history(config, run_id)

        # Step 10: Fail early if history is still missing
        if history is None:
            raise ValueError(
                f"\n\n❌  Error from experiment.py at _run_single_pipeline_entry()!\n"
                f"→ Training history is missing for run_id '{run_id}' — cannot extract metrics.\n\n"
            )

        # Step 11: Extract training/validation metrics
        metrics = extract_history_metrics(history)

        # Step 12: Define and verify best checkpoint path
        best_model_path = config.CHECKPOINT_PATH / run_id / "best.keras"
        if not best_model_path.exists():
            raise FileNotFoundError(
                f"\n\n❌  Error from experiment.py at _run_single_pipeline_entry()!\n"
                f"→ Best model file not found:\n→ {best_model_path}\n\n"
            )

        # Step 13: Load best model from checkpoint
        try:
            trained_model = load_model(best_model_path)
            trained_model._loaded_weights_mtime = os.path.getmtime(best_model_path)
            print(f"\n📥  Restored best model from:\n→ {best_model_path}")
        except Exception as e:
            raise RuntimeError(
                f"\n\n❌  Error from experiment.py at _run_single_pipeline_entry()!\n"
                f"→ Failed to load best model:\n→ {best_model_path}\n→ {e}\n\n"
            )

        # Step 14: Evaluate the model on test set
        metrics = evaluate_model(
            trained_model, history, test_data, test_labels, config, run_id
        )

        for k in ["final_test_loss", "final_test_acc"]:
            if k not in metrics:
                raise KeyError(
                    f"\n\n❌  Error from experiment.py at _run_single_pipeline_entry()!\n"
                    f"→ evaluate_model() result is missing key: '{k}'\n\n"
                )

        final_test_loss = metrics["final_test_loss"]
        final_test_acc = metrics["final_test_acc"]

        # Step 15: Build structured result dictionary
        evaluation = _create_evaluation_dictionary(
            model_number, run, config_name,
            time.time() - start_time, config,
            metrics, final_test_loss, final_test_acc
        )

        # Step 16: Print and append experiment results
        print("\n📊  Dumping experiment results:")
        print(json.dumps([to_json_compatible(evaluation)], indent=2))

        all_results.append(evaluation)
        with open(result_file, "w") as jf:
            json.dump(to_json_compatible(all_results), jf, indent=2)

        # Step 17: Confirm experiment completion
        print(f"\n✅  m{model_number} run {run} with '{config_name}' successfully executed")

    except Exception as e:
        # Step 18: Log structured error record
        log_to_json(
            config.ERROR_PATH,
            key=run_id,
            record={
                "model": model_number,
                "run": run,
                "config_name": config_name,
                "error": str(e),
                "exception_type": type(e).__name__,
                "trace": traceback.format_exc()
            },
            error=True
        )
        raise

    # Step 19: Return updated result file path
    return result_file


def run_pipeline(pipeline):
    """
    Orchestrates the full experiment queue based on the pipeline list.
    Executes each run and saves consolidated results at the end.
    """

    # Step 0: Print function execution header
    print("\n🎯  run_pipeline is executing ...")

    # Step 1: Validate pipeline format
    if not isinstance(pipeline, list) or not all(isinstance(entry, tuple) and len(entry) == 2 for entry in pipeline):
        raise TypeError(
            f"\n\n❌  Error from experiment.py at run_pipeline()!\n"
            f"→ 'pipeline' must be a list of (model_number, config_name) tuples.\n→ Got: {pipeline}\n\n"
        )

    # Step 2: Validate CONFIG.CONFIG_PATH
    if not isinstance(CONFIG.CONFIG_PATH, Path):
        raise TypeError(
            f"\n\n❌  Error from experiment.py at run_pipeline()!\n"
            f"→ CONFIG.CONFIG_PATH must be a pathlib.Path\n→ Got: {type(CONFIG.CONFIG_PATH)}\n\n"
        )

    # Step 3: Generate Tehran timestamp and initialize logging
    tehran_tz = datetime.timezone(datetime.timedelta(hours=3, minutes=30))
    timestamp = datetime.datetime.now(tehran_tz).strftime("%Y-%m-%d_%H-%M-%S")
    print(f"\n⏱️  Timestamp for this session: {timestamp}")
    log_stream, result_file, all_results = initialize_logging(timestamp)

    try:
        # Step 4: Load previous results and prepare counters
        completed_triplets = load_previous_results(result_file, all_results)
        model_run_counter = {}

        # Step 5: Run each pipeline entry
        for i, (model_number, config_name) in enumerate(pipeline):
            print(f"\n⚙️  Piplining experiment {i+1}/{len(pipeline)}")

            config_path = CONFIG.CONFIG_PATH / f"{config_name}.json"
            model_run_counter[model_number] = model_run_counter.get(model_number, 0) + 1
            run = model_run_counter[model_number]

            _run_single_pipeline_entry(
                model_number=model_number,
                config_path=config_path,
                config_name=config_name,
                run=run,
                timestamp=timestamp,
                completed_triplets=completed_triplets,
                all_results=all_results,
                result_file=result_file
            )

        # Step 6: Final result logging
        print(f"\n📦   Completed {len(all_results)} total experiment runs")
        try:
            with open(result_file, "w") as jf:
                json.dump(to_json_compatible(all_results), jf, indent=2)
        except Exception as e:
            raise IOError(
                f"\n\n❌  Error from experiment.py at run_pipeline()!\n"
                f"→ Failed to write consolidated result file:\n→ {result_file}\n→ {e}\n\n"
            )

    finally:
        # Step 7: Cleanup and restore standard streams
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        if log_stream:
            log_stream.close()


# Print module successfully executed
print("\n✅  experiment.py successfully executed.")
