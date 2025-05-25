# Import standard libraries
import random

# Import third-party libraries
import numpy as np
from PIL import Image
from torchvision import datasets, transforms


# CIFAR-10 mean/std (for normalization)
_CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]
_CIFAR10_STD  = [0.2023, 0.1994, 0.2010]

# Cutout transform: masks a square region of the image to improve robustness
class Cutout:
    def __init__(self, size=16):
        # Size of the square mask (in pixels)
        self.size = size

    def __call__(self, img):
        # Convert PIL image to NumPy array if needed
        if isinstance(img, Image.Image):
            img = np.array(img)

        # Get image height and width
        h, w = img.shape[:2]

        # Random center for the cutout mask
        y = random.randint(0, h)
        x = random.randint(0, w)

        # Compute mask boundaries
        y1 = max(0, y - self.size // 2)
        y2 = min(h, y + self.size // 2)
        x1 = max(0, x - self.size // 2)
        x2 = min(w, x + self.size // 2)

        # Apply zero mask to the selected region (across all channels)
        img[y1:y2, x1:x2, :] = 0

        # Convert back to PIL image for compatibility with torchvision
        return Image.fromarray(img)


# Function to normalize dataset
def build_normalization_transform():
    """
    Returns a torchvision transform for CIFAR-10 normalization.

    Converts input to tensor and normalizes using CIFAR-10 mean/std.
    """

    # Print header for function execution
    print("\n🎯  build_normalization_transform")


    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=_CIFAR10_MEAN, std=_CIFAR10_STD)
    ])


# Function to augment dataset
def build_augmentation_transform(config):
    """
    Constructs a transform pipeline with augmentations + normalization for CIFAR-10.

    Applies random crop, flip, and cutout if enabled in the config,
    followed by standard normalization.

    Args:
        config (Config): Configuration with AUGMENT_MODE settings.

    Returns:
        torchvision.transforms.Compose: Augmented and normalized transform pipeline.
    """

    # Print header for function execution
    print("\n🎯  build_augmentation_transform")

    # Ensure config has AUGMENT_MODE field
    assert hasattr(config, "AUGMENT_MODE"), "Missing AUGMENT_MODE in config"

    # Start with conversion to PIL format (required for torchvision transforms)
    ops = [transforms.ToPILImage()]

    # Access augmentation config
    augment = config.AUGMENT_MODE

    # Conditionally add augmentations if enabled
    if augment.get("enabled", False):
        if augment.get("random_crop", False):
            ops.append(transforms.RandomCrop(32, padding=4))  # Randomly crop with padding
        if augment.get("random_flip", False):
            ops.append(transforms.RandomHorizontalFlip())     # Random horizontal flip
        if augment.get("cutout", False):
            ops.append(Cutout(size=16))                       # Apply Cutout for occlusion

    # Always append normalization steps (ToTensor + Normalize)
    ops += build_normalization_transform().transforms

    # Return the composed transform pipeline
    return transforms.Compose(ops)




# Function to load, optionally augment, and always standardize CIFAR-10
def build_dataset(config):
    """
    Loads the CIFAR-10 dataset and applies preprocessing.

    Supports:
    - Optional light mode (subset of CIFAR-10)
    - Optional data augmentation via flags:
        - random_crop
        - random_flip
        - cutout
    - Per-channel normalization with CIFAR-10 mean/std.

    Args:
        config (Config): Configuration object with LIGHT_MODE and AUGMENT_MODE dict.

    Returns:
        tuple: (train_data, train_labels, test_data, test_labels)
            All data arrays are np.float32 with shape (N, 32, 32, 3)
    """

    # Print header for function execution
    print("\n🎯  build_dataset")

    # Load CIFAR-10
    train_set = datasets.CIFAR10(root=config.DATA_PATH, train=True, download=True)
    test_set = datasets.CIFAR10(root=config.DATA_PATH, train=False, download=True)

    train_images = train_set.data
    test_images = test_set.data
    train_labels = np.array(train_set.targets)
    test_labels = np.array(test_set.targets)

    # Optional light mode slicing
    if config.LIGHT_MODE:
        train_images, train_labels = train_images[:5000], train_labels[:5000]
        test_images, test_labels = test_images[:1000], test_labels[:1000]
    else:
        train_images, train_labels = train_images[:-5000], train_labels[:-5000]

    # Build transforms for training and test
    train_transform = build_augmentation_transform(config)
    test_transform = build_normalization_transform()

    # Apply transforms and convert to float32 NumPy arrays
    train_data = [train_transform(img).permute(1, 2, 0).numpy() for img in train_images]
    test_data = [test_transform(img).permute(1, 2, 0).numpy() for img in test_images]

    return (
        np.stack(train_data).astype(np.float32),
        train_labels,
        np.stack(test_data).astype(np.float32),
        test_labels,
    )


# Print module successfully executed
print("\n✅  data.py successfully executed")
