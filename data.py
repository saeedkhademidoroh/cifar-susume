# Import third-party libraries
import numpy as np
from torchvision import datasets, transforms


# CIFAR-10 mean/std (for normalization)
_CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]
_CIFAR10_STD  = [0.2023, 0.1994, 0.2010]

# Function to load and preprocess CIFAR-10 based on model_number
def load_dataset(model_number: int, config):
    """
    Function to load and preprocess the CIFAR-10 dataset based on the model_number.

    Applies variant-specific preprocessing and augmentation depending on the model.
    Supports LIGHT_MODE subsampling and AUGMENT_MODE augmentation toggle.

    Args:
        model_number (int): Identifier for architecture variant
        config (Config): Configuration object with toggles and path

    Returns:
        tuple: (train_data, train_labels, test_data, test_labels)
    """

    # Print header for function execution
    print("\n🎯  load_dataset")

    # Load CIFAR-10 dataset
    train_set = datasets.CIFAR10(root=config.DATA_PATH, train=True, download=True)
    test_set = datasets.CIFAR10(root=config.DATA_PATH, train=False, download=True)

    train_images = train_set.data
    test_images = test_set.data
    train_labels = np.array(train_set.targets)
    test_labels = np.array(test_set.targets)

    if config.LIGHT_MODE:
        train_images = train_images[:1000]
        train_labels = train_labels[:1000]
        test_images = test_images[:200]
        test_labels = test_labels[:200]
    else:
        train_images = train_images[:-5000]
        train_labels = train_labels[:-5000]

    # Models 0–5: Rescale to [0,1] + optional augmentation
    if model_number in [0, 1, 2, 3, 4, 5]:
        train_data = train_images.astype(np.float32) / 255.0
        if config.AUGMENT_MODE:
            train_data = _augment_dataset(train_data)

    # Models 6–7: Standardize + optional augmentation
    elif model_number in [6, 7]:
        train_data = _standardize(train_images)
        if config.AUGMENT_MODE:
            train_data = _augment_dataset(train_data)

    # Model 8
    elif model_number == 8:
        if config.AUGMENT_MODE:
            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=_CIFAR10_MEAN, std=_CIFAR10_STD)
            ])
            augmented = [transform(img).permute(1, 2, 0).numpy() for img in train_images]
            train_data = np.stack(augmented, axis=0)
        else:
            train_data = _standardize(train_images)

    else:
        raise ValueError(f"❌  ValueError from data.py in load_dataset():\nmodel_number={model_number}\n")

    # Standardize test data (always)
    test_data = test_images.astype(np.float32) / 255.0
    for i in range(3):
        test_data[..., i] = (test_data[..., i] - _CIFAR10_MEAN[i]) / _CIFAR10_STD[i]

    return train_data, train_labels, test_data, test_labels


# Function to augment dataset
def _augment_dataset(images):
    """
    Applies CIFAR-10 style augmentation using torchvision.transforms.

    Augmentations:
    - Random crop with padding
    - Random horizontal flip
    - Color jitter (brightness and contrast)

    Args:
        images (np.ndarray): Array of images (float32 in [0, 1], shape: [N, 32, 32, 3])

    Returns:
        np.ndarray: Augmented images (same shape)
    """

    # Print header for function execution
    print("\n🎯  _augment_dataset")

    # Define CIFAR-10 augmentation pipeline
    transform = transforms.Compose([
        transforms.ToPILImage(),                              # Convert to PIL for torchvision compatibility
        transforms.RandomCrop(32, padding=4),                 # Random crop with padding
        transforms.RandomHorizontalFlip(),                    # Random horizontal flip
        transforms.ColorJitter(brightness=0.1, contrast=0.1), # Slight brightness and contrast jitter
        transforms.ToTensor()                                 # Convert back to tensor [C, H, W]
    ])

    # Apply transform to each image and permute from [C, H, W] to [H, W, C], then convert to NumPy
    augmented = [transform(img).numpy().transpose(1, 2, 0) for img in images]

    # Stack list of transformed images into a single NumPy array with shape (N, 32, 32, 3)
    return np.stack(augmented, axis=0)



# Function to standardize CIFAR-10 input images using global dataset statistics
def _standardize(images):
    """
    Normalize image pixels to [0, 1] then standardize each RGB channel
    using CIFAR-10 mean and standard deviation.

    Args:
        images (np.ndarray): Image array in uint8 format, shape (N, 32, 32, 3)

    Returns:
        np.ndarray: Standardized float32 array of the same shape
    """

    # Print header to indicate function execution
    print("\n🎯  _standardize")

    # Convert uint8 images (0–255) to float32 and rescale to [0.0, 1.0]
    images = images.astype(np.float32) / 255.0

    # Apply per-channel standardization: subtract mean and divide by std
    for i in range(3):
        images[..., i] = (images[..., i] - _CIFAR10_MEAN[i]) / _CIFAR10_STD[i]

    # Return standardized image array
    return images


# Print module successfully executed
print("\n✅  data.py successfully executed")
