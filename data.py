# Import third-party libraries
import numpy as np
from torchvision import datasets, transforms
from config import CONFIG

# Function to load raw CIFAR-10 train/test data and labels
def load_dataset(data_dir=CONFIG.DATA_PATH):
    """
    Loads CIFAR-10 dataset and returns raw train/test data and labels.

    Returns:
        tuple: train_data (np.ndarray), train_labels (np.ndarray),
               test_data (np.ndarray), test_labels (np.ndarray)
    """
    # Load without transform to access raw NumPy arrays
    train_set = datasets.CIFAR10(root=data_dir, train=True, download=True)
    test_set = datasets.CIFAR10(root=data_dir, train=False, download=True)

    train_data = np.array(train_set.data)
    train_labels = np.array(train_set.targets)
    test_data = np.array(test_set.data)
    test_labels = np.array(test_set.targets)

    return train_data, train_labels, test_data, test_labels

# Print confirmation message
print("\n✅ data.py successfully executed")
