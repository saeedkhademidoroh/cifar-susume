# Import third-party libraries
import numpy as np
import torch
from torchvision import datasets
from config import CONFIG

# Function to load normalized CIFAR-10 data with one-hot labels (optional)
def load_dataset(data_dir=CONFIG.DATA_PATH, one_hot=False):
    """
    Loads CIFAR-10 dataset, normalizes images to [0,1], and optionally one-hot encodes labels.

    Args:
        data_dir (Path): Directory to store CIFAR-10
        one_hot (bool): Whether to return one-hot encoded labels (as torch tensors)

    Returns:
        tuple: train_data, train_labels, test_data, test_labels
    """
    train_set = datasets.CIFAR10(root=data_dir, train=True, download=True)
    test_set = datasets.CIFAR10(root=data_dir, train=False, download=True)

    # Normalize images
    train_data = train_set.data.astype(np.float32) / 255.0
    test_data = test_set.data.astype(np.float32) / 255.0

    train_labels = np.array(train_set.targets)
    test_labels = np.array(test_set.targets)

    if one_hot:
        train_labels = torch.nn.functional.one_hot(torch.tensor(train_labels), num_classes=10).numpy()
        test_labels = torch.nn.functional.one_hot(torch.tensor(test_labels), num_classes=10).numpy()

    # train_data, train_labels, test_data, test_labels = load_dataset(one_hot=True)
    return train_data, train_labels, test_data, test_labels


# Print confirmation message
print("\n✅ data.py successfully executed\n")
