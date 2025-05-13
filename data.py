# Import third-party libraries
import numpy as np
from torchvision import datasets

# Import project-specific libraries
from config import CONFIG


# Function to load dataset
def load_dataset_m2():
    return load_dataset_m0()

# Function to load dataset
def load_dataset_m1():
    return load_dataset_m0()


# Function to load dataset
def load_dataset_m0():
    """
    Loads CIFAR-10, normalizes pixel values, and optionally trims for LIGHT_MODE.

    Args:
        path (Path): Directory to store or load CIFAR-10.

    Returns:
        tuple: (train_data, train_labels, test_data, test_labels)
    """

    # Print header for function execution
    print(f"\n🎯 load_dataset_m0\n")

    # Load CIFAR-10 training and test sets
    train_set = datasets.CIFAR10(root=CONFIG.DATA_PATH, train=True, download=True)
    test_set = datasets.CIFAR10(root=CONFIG.DATA_PATH, train=False, download=True)

    # Normalize image pixel values to range [0, 1]
    train_data = train_set.data.astype(np.float32) / 255.0
    test_data = test_set.data.astype(np.float32) / 255.0

    # Convert labels to numpy arrays
    train_labels = np.array(train_set.targets)
    test_labels = np.array(test_set.targets)

    # Reduce dataset size for local/light runs
    if CONFIG.LIGHT_MODE:
        train_data = train_data[:1000]
        train_labels = train_labels[:1000]
        test_data = test_data[:200]
        test_labels = test_labels[:200]
    else:
        # Use all but last 5000 as training
        train_data = train_data[:-5000]
        train_labels = train_labels[:-5000]

    # Return dataset split as train and test
    return train_data, train_labels, test_data, test_labels


# Function to dispatch dataset loaders
def load_dataset(model_number):
    """
    Routes dataset loading based on model number.

    Args:
        model_number (int): Model variant identifier (e.g., 1, 2)

    Returns:
        tuple: (train_data, train_labels, test_data, test_labels)
    """

    # Print header for function execution
    print(f"\n🎯 load_dataset")

    # Dispatch dataset loader
    try:
        # Construct function name and resolve dynamically
        loader_fn = globals()[f"load_dataset_m{model_number}"]
        return loader_fn()
    except KeyError:
        raise ValueError(f"❌ ValueError:\nmodel_number={model_number}\n")



# Print confirmation message
print("\n✅ data.py successfully executed")
