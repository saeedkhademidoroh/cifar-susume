# Standard libraries
import os
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from config import CONFIG
from data import load_dataset
from log import log_to_json

# ------------------ VGG-style Small CNN (under 400K params) ------------------ #

class MiniVGG(nn.Module):
    def __init__(self):
        super(MiniVGG, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),  # 3x32x32 → 16x32x32
            nn.ReLU(),
            nn.MaxPool2d(2, 2),              # → 16x16x16
            nn.Conv2d(16, 32, 3, padding=1), # → 32x16x16
            nn.ReLU(),
            nn.MaxPool2d(2, 2),              # → 32x8x8
        )
        self.classifier = nn.Sequential(
            nn.Linear(32 * 8 * 8, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

# ----------------------------- Config & Setup ----------------------------- #

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 20
CHECKPOINT_DIR = "cifar/checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ------------------------- Load and Prepare Data ------------------------- #

train_data, train_labels, test_data, test_labels = load_dataset()
train_data, val_data, train_labels, val_labels = train_test_split(
    train_data, train_labels, test_size=0.1, random_state=42
)

def to_loader(x, y, shuffle=False):
    x = torch.tensor(x).permute(0, 3, 1, 2).float()  # NHWC → NCHW
    y = torch.tensor(y).long()
    return DataLoader(TensorDataset(x, y), batch_size=CONFIG.BATCH_SIZE, shuffle=shuffle, num_workers=CONFIG.NUM_WORKERS)

train_loader = to_loader(train_data, train_labels, shuffle=True)
val_loader = to_loader(val_data, val_labels)
test_loader = to_loader(test_data, test_labels)

# ------------------------- Model, Loss, Optimizer ------------------------ #

model = MiniVGG().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()
best_val_acc = 0.0

# ------------------------------- Training ------------------------------- #

for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        _, preds = outputs.max(1)
        correct += preds.eq(y).sum().item()
        total += y.size(0)

    train_acc = correct / total
    avg_loss = total_loss / len(train_loader)

    # --------------------------- Validation --------------------------- #

    model.eval()
    val_correct, val_total = 0, 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            preds = model(x).argmax(1)
            val_correct += preds.eq(y).sum().item()
            val_total += y.size(0)

    val_acc = val_correct / val_total

    # ----------------------- Checkpoint + Log ------------------------ #

    is_best = val_acc > best_val_acc
    if is_best:
        best_val_acc = val_acc
        ckpt_path = f"{CHECKPOINT_DIR}/best_model.pth"
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, ckpt_path)

        log_to_json(key="checkpoints", record={
            "path": str(ckpt_path),
            "epoch": epoch,
            "val_accuracy": round(val_acc, 4),
            "train_accuracy": round(train_acc, 4),
            "loss": round(avg_loss, 4),
            "status": "best"
        })

    print(f"Epoch {epoch:02d} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | Loss: {avg_loss:.4f}")


# Print confirmation message
print("\n✅ train.py successfully executed")