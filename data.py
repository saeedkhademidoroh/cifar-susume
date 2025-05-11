# Import third-party libraries
import numpy as np
from torchvision import datasets

# Import project-specific libraries
from config import CONFIG


# Function to load dataset for model variant m2
def load_dataset_m2():
    # Currently identical to m1 but can be customized later
    return load_dataset_m1()


# Function to load and preprocess CIFAR-10 for model variant m1
def load_dataset_m1():
    """
    Loads CIFAR-10, normalizes pixel values, and optionally trims for LIGHT_MODE.

    Args:
        path (Path): Directory to store or load CIFAR-10.

    Returns:
        tuple: (train_data, train_labels, test_data, test_labels)
    """


    # Load CIFAR-10 training and test sets
    train_set = datasets.CIFAR10(root=CONFIG.DATA_PATH, train=True, download=True)
    test_set = datasets.CIFAR10(root=CONFIG.DATA_PATH, train=False, download=True)

    # Normalize image pixel values to range [0, 1]
    train_data = train_set.data.astype(np.float32) / 255.0
    test_data = test_set.data.astype(np.float32) / 255.0

    # Convert labels to NumPy arrays
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

    return train_data, train_labels, test_data, test_labels


# Routing function to dispatch model-specific dataset loaders
def load_dataset(model_number):
    """
    Routes dataset loading based on model number.

    Args:
        model_number (int): Model variant identifier (e.g., 1, 2)

    Returns:
        tuple: (train_data, train_labels, test_data, test_labels)
    """
    print(f"\n🎯 load_dataset_m{model_number}")

    if model_number == 1:
        return load_dataset_m1()
    elif model_number == 2:
        return load_dataset_m2()
    else:
        raise ValueError(f"❌ ValueError:\nmodel_number={model_number}\n")


# Confirmation message on successful module execution
print("\n✅ data.py successfully executed\n")
