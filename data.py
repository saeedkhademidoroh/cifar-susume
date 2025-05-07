# Import third-party libraries
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

# Import project-specific modules
from config import DATA_DIR, BATCH_SIZE, NUM_WORKERS

# Function to prepare CIFAR-10 dataloaders
def load_dataset(data_dir=DATA_DIR, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS):
    """
    Loads CIFAR-10 dataset and returns train/val/test loaders.

    Args:
        data_dir (Path): Directory for CIFAR-10 files
        batch_size (int): Batch size for loaders
        num_workers (int): Data loading worker threads

    Returns:
        tuple: train_loader, val_loader, test_loader
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    full_train = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)
    train_set = Subset(full_train, range(0, 45000))
    val_set = Subset(full_train, range(45000, 50000))
    test_set = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader

# Print confirmation message
print("\n✅ data.py successfully executed")
