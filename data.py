# Import standard libraries
import random

# Import third-party libraries
import numpy as np
from PIL import Image
from torchvision import transforms, datasets


# CIFAR-10 per-channel mean and standard deviation (used for normalization)
_CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]
_CIFAR10_STD  = [0.2023, 0.1994, 0.2010]

# Print verification log
print(f"\n📊  CIFAR-10 normalization constants loaded:")
print(f"→ Mean per channel: {_CIFAR10_MEAN}")
print(f"→ Std per channel: {_CIFAR10_STD}")

# Cutout transform: masks a square region of the image to improve robustness
class Cutout:
    def __init__(self, size=16):
        """
        Initializes the Cutout transformation.

        Args:
            size (int): Side length of the square mask to apply.
        """

        # Store size of the square mask in pixels
        self.size = size

        # Print initialization log
        print(f"\n🧊  Cutout transformation initialized successfully:")
        print(f"→ Active mask size: {self.size}×{self.size} pixels")

    def __call__(self, img):
        """
        Applies the Cutout mask to the input image.

        Args:
            img (PIL.Image or np.ndarray): Input image.

        Returns:
            PIL.Image: Image with a square region zeroed out.
        """

        # Convert PIL image to NumPy array if needed
        if isinstance(img, Image.Image):
            img = np.array(img)

        # Print input shape for debugging
        print(f"\n🧊  Cutout input shape:\n→ {img.shape} (H×W×C)")

        # Get image height and width
        h, w = img.shape[:2]

        # Random center coordinates for the mask
        y = random.randint(0, h)
        x = random.randint(0, w)

        # Compute mask boundaries
        y1 = max(0, y - self.size // 2)
        y2 = min(h, y + self.size // 2)
        x1 = max(0, x - self.size // 2)
        x2 = min(w, x + self.size // 2)

        # Ensure bounds are valid
        assert y1 >= 0 and y2 <= h and x1 >= 0 and x2 <= w, "Cutout mask out of image bounds"

        # Apply a black square (zero mask) across all RGB channels
        img[y1:y2, x1:x2, :] = 0

        # Calculate number of masked pixels (ignores channel redundancy)
        masked_pixels = (y2 - y1) * (x2 - x1)

        # Extract pixel values at center of mask
        center_y, center_x = (y1 + y2) // 2, (x1 + x2) // 2
        sample_pixel = img[center_y, center_x]

        # Check whether all masked pixels are zeroed across all channels
        all_zero = np.all(img[y1:y2, x1:x2] == 0)

        print(f"\n🧊  Cutout mask applied to image successfully:")
        print(f"→ Masked area: {y2 - y1}×{x2 - x1} pixels")
        print(f"→ Total masked pixels: {masked_pixels}")
        print(f"→ Mask center pixel value: {sample_pixel}")
        print(f"→ All masked pixels fully zeroed: {all_zero}")
        print(f"→ Post-cutout pixel range: min={img.min()}, max={img.max()}")

        # Return the image as PIL format for downstream compatibility
        return Image.fromarray(img)


# Function to build normalization transform
def build_normalization_transform():
    """
    Returns a torchvision transform for CIFAR-10 normalization.

    Converts input to tensor and normalizes using CIFAR-10 mean/std.
    """

    # Print header for function execution
    print("\n🎯  build_normalization_transform is executing ...")

    # Log normalization details
    print("\n📊  Applying normalization to dataset:")
    print(f"→ Normalization mean: {_CIFAR10_MEAN}")
    print(f"→ Normalization std:  {_CIFAR10_STD}")

    # Return a composed transform that converts to tensor and normalizes using CIFAR-10 stats
    return transforms.Compose([
        transforms.ToTensor(),  # Converts H×W×C image to C×H×W tensor in [0, 1]
        transforms.Normalize(mean=_CIFAR10_MEAN, std=_CIFAR10_STD)  # Normalizes using fixed per-channel stats
    ])


# Function to build augmentation transform
def _build_augmentation_transform(config):
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
    print("\n🎯  build_augmentation_transform is executing ...")

    # Ensure config has AUGMENT_MODE field
    if not hasattr(config, "AUGMENT_MODE"):
        raise ValueError("\n\n❌  ValueError from data.py at build_augmentation_transform()!\nMissing 'AUGMENT_MODE' in config\n\n")

    # Access augmentation config
    augment = config.AUGMENT_MODE

    # Print augmentation policy summary
    print("\n🎛️  Augmentation policy is being configured:")
    print(f"→ random_crop enabled:  {augment.get('random_crop', False)}")
    print(f"→ random_flip enabled:  {augment.get('random_flip', False)}")
    print(f"→ cutout enabled:       {augment.get('cutout', False)}")

    # Start with conversion to PIL format (required for torchvision transforms)
    ops = [transforms.ToPILImage()]

    # Conditionally add augmentations if enabled
    if augment.get("enabled", False):
        if augment.get("random_crop", False):
            ops.append(transforms.RandomCrop(32, padding=4))  # Randomly crop with padding
        if augment.get("random_flip", False):
            ops.append(transforms.RandomHorizontalFlip())     # Random horizontal flip
        if augment.get("cutout", False):
            ops.append(Cutout(size=16))                       # Apply Cutout for occlusion

    # Append normalization transforms (ToTensor + Normalize)
    ops += build_normalization_transform().transforms

    # Return composed pipeline with augmentations and normalization
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
    print("\n🎯  build_dataset is executing ...")

    # Load CIFAR-10 from torchvision.datasets
    train_set = datasets.CIFAR10(root=config.DATA_PATH, train=True, download=True)
    test_set = datasets.CIFAR10(root=config.DATA_PATH, train=False, download=True)

    train_images = train_set.data
    test_images = test_set.data
    train_labels = np.array(train_set.targets)
    test_labels = np.array(test_set.targets)

    # Apply light mode (smaller subset) if enabled
    if config.LIGHT_MODE:
        train_images, train_labels = train_images[:5000], train_labels[:5000]
        test_images, test_labels = test_images[:1000], test_labels[:1000]
    else:
        train_images, train_labels = train_images[:-5000], train_labels[:-5000]

    # Build transform pipelines
    train_transform = _build_augmentation_transform(config)
    test_transform = build_normalization_transform()

    # Apply training transforms and verify augmentation
    train_data = []
    for i, img in enumerate(train_images):
        if i == 0:
            print(f"\n🎛️  First training image is being transformed:")
            print(f"→ Original shape: {img.shape}")
            print(f"→ Original pixel range: min={img.min()}, max={img.max()}")

        transformed = train_transform(img)

        if i == 0:
            arr = transformed.permute(1, 2, 0).numpy()
            print(f"→ Transformed shape: {arr.shape}")
            print(f"→ Transformed pixel range: min={arr.min():.3f}, max={arr.max():.3f}")

        train_data.append(transformed.permute(1, 2, 0).numpy())

    # Apply test transforms (no augmentation)
    test_data = [test_transform(img).permute(1, 2, 0).numpy() for img in test_images]

    # Return final arrays and labels
    return (
        np.stack(train_data).astype(np.float32),
        train_labels,
        np.stack(test_data).astype(np.float32),
        test_labels,
    )


# Print module successfully executed
print("\n✅  data.py successfully executed.")
