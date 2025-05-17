# Import third-party libraries
import numpy as np
from torchvision import datasets


# Function to load dataset for model 5
def _load_dataset_m5(config):
    return _load_dataset_m0(config)


# Function to load dataset for model 4
def _load_dataset_m4(config):
    return _load_dataset_m0(config)


# Function to load dataset for model 3
def _load_dataset_m3(config):
    return _load_dataset_m0(config)


# Function to load dataset for model 2
def _load_dataset_m2(config):
    return _load_dataset_m0(config)


# Function to load dataset for model 1
def _load_dataset_m1(config):
    return _load_dataset_m0(config)


# Function to load dataset for model 0 (shared logic)
def _load_dataset_m0(config):
    """
    Function to load and preprocess the CIFAR-10 dataset.

    Loads CIFAR-10 from the specified data path, normalizes pixel values,
    and trims the dataset based on LIGHT_MODE flag.

    Args:
        config (Config): Configuration object containing DATA_PATH and LIGHT_MODE

    Returns:
        tuple: (train_data, train_labels, test_data, test_labels)
    """

    # Print header for function execution
    print(f"\n🎯 _load_dataset_m0")

    # Download CIFAR-10 dataset
    train_set = datasets.CIFAR10(root=config.DATA_PATH, train=True, download=True)
    test_set = datasets.CIFAR10(root=config.DATA_PATH, train=False, download=True)

    # Normalize pixel values to [0, 1]
    train_data = train_set.data.astype(np.float32) / 255.0
    test_data = test_set.data.astype(np.float32) / 255.0

    # Convert labels to numpy arrays
    train_labels = np.array(train_set.targets)
    test_labels = np.array(test_set.targets)

    # Subsample if LIGHT_MODE is enabled
    if config.LIGHT_MODE:
        train_data = train_data[:1000]
        train_labels = train_labels[:1000]
        test_data = test_data[:200]
        test_labels = test_labels[:200]
    else:
        # Exclude last 5000 training samples
        train_data = train_data[:-5000]
        train_labels = train_labels[:-5000]

    # Return processed dataset
    return train_data, train_labels, test_data, test_labels


# Function to dispatch dataset loader by model number
def dispatch_load_dataset(model_number, config):
    """
    Function to route dataset loading based on the model variant number.

    Dynamically constructs the appropriate function name (e.g., load_dataset_m2)
    and invokes it using the global namespace. Raises ValueError if the function
    is not defined.

    Args:
        model_number (int): Model identifier (e.g., 0 to 5)
        config (Config): Configuration object to pass to the loader

    Returns:
        tuple: (train_data, train_labels, test_data, test_labels)
    """

    # Print header for function execution
    print("\n🎯 dispatch_load_dataset")

    try:
        # Dynamically resolve the dataset loader function by model number
        loader_fn = globals()[f"_load_dataset_m{model_number}"]
        return loader_fn(config)  # Return loaded dataset from resolved function

    except KeyError:
        # Raise clear error if the loader function is not defined
        raise ValueError(f"❌ ValueError:\nmodel_number={model_number}\n")


# Print module successfully executed
print("\n✅ data.py successfully executed")
