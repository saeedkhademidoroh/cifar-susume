# Standard library imports
import os
import json
from pathlib import Path

# Third-party imports
from keras.api.callbacks import Callback, ModelCheckpoint
from keras.api.models import load_model

# Project-specific imports
from config import CONFIG
from log import log_to_json

# Custom callback to save after every epoch
class RecoveryCheckpoint(Callback):
    def __init__(self, ckpt_dir: Path):
        super().__init__()
        self.ckpt_dir = ckpt_dir
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.model_path = ckpt_dir / "latest.keras"
        self.state_path = ckpt_dir / "state.json"

    def on_epoch_end(self, epoch, logs=None):
        self.model.save(self.model_path)
        with open(self.state_path, "w") as f:
            json.dump({"initial_epoch": epoch + 1}, f)
        print(f"💾 Saved recovery checkpoint (epoch {epoch + 1})")


# Function to prepare ModelCheckpoint callbacks
def get_checkpoint_callbacks(model_ckpt_dir: Path, verbose: int = 1):
    best_model_path = model_ckpt_dir / "best.keras"
    per_epoch_path = model_ckpt_dir / "epoch_{epoch:02d}.keras"

    return [
        ModelCheckpoint(
            filepath=best_model_path,
            monitor='val_accuracy',
            save_best_only=True,
            save_weights_only=False,
            verbose=verbose
        ),
        ModelCheckpoint(
            filepath=per_epoch_path,
            save_best_only=False,
            save_weights_only=False,
            verbose=0
        )
    ]

# Resume checkpoint loader
def load_training_state(model_ckpt_dir: Path):
    state_path = model_ckpt_dir / "state.json"
    model_path = model_ckpt_dir / "latest.keras"
    if model_path.exists() and state_path.exists():
        with open(state_path, "r") as f:
            state = json.load(f)
        model = load_model(model_path)
        return model, state.get("initial_epoch", 0)
    return None, 0

# Training entry
def train_model(train_data, train_labels, model, model_name="mobilenet", verbose=1, result_file_path=None):
    print("\n🎯 Train Model 🎯")

    model_ckpt_dir = CONFIG.CHECKPOINT_PATH / model_name
    model_ckpt_dir.mkdir(parents=True, exist_ok=True)

    resumed_model, initial_epoch = load_training_state(model_ckpt_dir)
    if resumed_model:
        print(f"🔁 Resuming training from epoch {initial_epoch}")
        model = resumed_model
    else:
        initial_epoch = 0

    print(f"🧪 Starting training for {model_name}")
    train_data = train_data[:5000]
    train_labels = train_labels[:5000]

    callbacks = get_checkpoint_callbacks(model_ckpt_dir, verbose)
    callbacks.append(RecoveryCheckpoint(model_ckpt_dir))

    history = model.fit(
        x=train_data,
        y=train_labels,
        epochs=CONFIG.EPOCHS_COUNT,
        batch_size=CONFIG.BATCH_SIZE,
        validation_split=CONFIG.VALIDATION_SPLIT,
        callbacks=callbacks,
        verbose=verbose,
        initial_epoch=initial_epoch
    )

    best_val_acc = max(history.history.get("val_accuracy", [0]))
    if result_file_path:
        log_to_json(file_path=result_file_path, key="checkpoints", record={
            "model": model_name,
            "path": str(model_ckpt_dir / "best.keras"),
            "val_accuracy": round(float(best_val_acc), 4),
            "epochs": len(history.history.get("loss", [])),
            "status": "best"
        })

    return model, history

# Print confirmation
print("\n✅ train.py successfully executed")
