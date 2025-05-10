# Import third-party libraries
import numpy as np
import torch
from torchvision import datasets

# Import project-specific libraries
from config import CONFIG


# Function to load normalized CIFAR-10 data with one-hot labels (optional)
def load_dataset(data_path=CONFIG.DATA_PATH, one_hot=False):
    """
    Loads CIFAR-10 dataset, normalizes images to [0,1], and optionally one-hot encodes labels.

    Args:
        data_path (Path): Directory to store CIFAR-10
        one_hot (bool): Whether to return one-hot encoded labels (as torch tensors)

    Returns:
        tuple: train_data, train_labels, test_data, test_labels
    """

    # Print header for function execution
    print("\n🎯 load_dataset")

    # Download and load CIFAR-10 training and test sets
    train_set = datasets.CIFAR10(root=data_path, train=True, download=True)
    test_set = datasets.CIFAR10(root=data_path, train=False, download=True)

    # Normalize image pixel values to the [0, 1] range
    train_data = train_set.data.astype(np.float32) / 255.0
    test_data = test_set.data.astype(np.float32) / 255.0

    # Extract class labels
    train_labels = np.array(train_set.targets)
    test_labels = np.array(test_set.targets)

    # Optionally one-hot encode labels (if required by model)
    if one_hot:
        train_labels = torch.nn.functional.one_hot(torch.tensor(train_labels), num_classes=10).numpy()
        test_labels = torch.nn.functional.one_hot(torch.tensor(test_labels), num_classes=10).numpy()

    # Return all preprocessed data and labels
    return train_data, train_labels, test_data, test_labels


# Confirm successful execution of this module
print("\n✅ data.py successfully executed\n")
