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

class Cutout:
    def __init__(self, size=16):
        """
        Initializes the Cutout transformation.

        Args:
            size (int): Side length of the square mask to apply.
        """

        # Step 0: Print header for function execution
        print("\n🎯  Cutout.__init__ is executing ...")


        # Step 1: Store the mask size
        self.size = size

        # Step 2: Print initialization info
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

        # Log only on the very first call
        if not hasattr(self, "_has_logged"):
            self._has_logged = True
            do_log = True
        else:
            do_log = False

        # Step 0: Print header for function execution (only if logging)
        if do_log:
            print("\n🎯  Cutout.__call__ is executing ...")

        # Step 1: Convert image to NumPy if needed
        if isinstance(img, Image.Image):
            img = np.array(img)

        # Step 2: Log image shape (only if logging)
        if do_log:
            print(f"\n🧊  Cutout input shape:\n→ {img.shape} (H×W×C)")

        # Step 3: Extract dimensions
        h, w = img.shape[:2]

        # Step 4: Randomly pick center of the mask
        y = random.randint(0, h)
        x = random.randint(0, w)

        # Step 5: Calculate mask boundaries
        y1 = max(0, y - self.size // 2)
        y2 = min(h, y + self.size // 2)
        x1 = max(0, x - self.size // 2)
        x2 = min(w, x + self.size // 2)

        # Step 6: Assert boundaries are valid
        assert y1 >= 0 and y2 <= h and x1 >= 0 and x2 <= w, "Cutout mask out of image bounds"

        # Step 7: Apply zero-mask to selected region (across all channels)
        img[y1:y2, x1:x2, :] = 0

        # Step 8: Measure masked area
        masked_pixels = (y2 - y1) * (x2 - x1)

        # Step 9: Sample and log center pixel
        center_y, center_x = (y1 + y2) // 2, (x1 + x2) // 2
        sample_pixel = img[center_y, center_x]

        # Step 10: Confirm that region is fully zeroed
        all_zero = np.all(img[y1:y2, x1:x2] == 0)

        # Step 11: Print post-cutout summary (only if logging)
        if do_log:
            print(f"\n🧊  Cutout mask applied to image successfully:")
            print(f"→ Masked area: {y2 - y1}×{x2 - x1} pixels")
            print(f"→ Total masked pixels: {masked_pixels}")
            print(f"→ Mask center pixel value: {sample_pixel}")
            print(f"→ All masked pixels fully zeroed: {all_zero}")
            print(f"→ Post-cutout pixel range: min={img.min()}, max={img.max()}")

        # Step 12: Return the transformed image in PIL format
        return Image.fromarray(img)


# Function to build normalization-only transform
def build_normalization_transform():
    """
    Returns a torchvision transform for CIFAR-10 normalization.

    Converts input to tensor and normalizes using CIFAR-10 mean/std.
    """

    # Step 0: Print header for function execution
    print("\n🎯  build_normalization_transform is executing ...")

    # Step 1: Log normalization parameters
    print("\n📊  Applying normalization to dataset:")
    print(f"→ Normalization mean: {_CIFAR10_MEAN}")
    print(f"→ Normalization std:  {_CIFAR10_STD}")

    # Step 2: Return transform pipeline (ToTensor + Normalize)
    return transforms.Compose([
        transforms.ToTensor(),  # Step 2.1: Convert H×W×C image to C×H×W in [0, 1]
        transforms.Normalize(mean=_CIFAR10_MEAN, std=_CIFAR10_STD)  # Step 2.2: Normalize using CIFAR-10 stats
    ])


# Function to build full augmentation + normalization transform
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

    # Step 0: Print header for function execution
    print("\n🎯  build_augmentation_transform is executing ...")

    # Step 1: Validate AUGMENT_MODE exists in config
    if not hasattr(config, "AUGMENT_MODE"):
        raise ValueError("\n\n❌  ValueError from data.py at build_augmentation_transform()!\nMissing 'AUGMENT_MODE' in config\n\n")

    # Step 2: Load augmentation settings
    augment = config.AUGMENT_MODE

    # Step 3: Log augmentation flags
    print("\n🎛️  Augmentation policy is being configured:")
    print(f"→ random_crop enabled:  {augment.get('random_crop', False)}")
    print(f"→ random_flip enabled:  {augment.get('random_flip', False)}")
    print(f"→ cutout enabled:       {augment.get('cutout', False)}")

    # Step 4: Initialize transform sequence
    ops = [transforms.ToPILImage()]  # Convert to PIL format first

    # Step 5: Conditionally append augmentations
    if augment.get("enabled", False):
        if augment.get("random_crop", False):
            ops.append(transforms.RandomCrop(32, padding=4))  # Step 5.1: Random crop
        if augment.get("random_flip", False):
            ops.append(transforms.RandomHorizontalFlip())     # Step 5.2: Random flip
        if augment.get("cutout", False):
            ops.append(Cutout(size=16))                       # Step 5.3: Cutout

    # Step 6: Add normalization steps from helper
    ops += build_normalization_transform().transforms

    # Step 7: Return composed transform pipeline
    return transforms.Compose(ops)


def build_dataset(config, val_split=5000):
    """
    Loads CIFAR-10, applies preprocessing, and splits train/val as needed.
    Returns: train_data, train_labels, val_data, val_labels, test_data, test_labels
    """

    # Step 0: Print header for function execution
    print("\n🎯  build_dataset is executing ...")

    # Step 1: Load CIFAR-10 dataset from torchvision
    train_set = datasets.CIFAR10(root=config.DATA_PATH, train=True, download=True)
    test_set = datasets.CIFAR10(root=config.DATA_PATH, train=False, download=True)

    # Step 2: Extract raw images and labels
    train_images = train_set.data
    train_labels = np.array(train_set.targets)
    test_images = test_set.data
    test_labels = np.array(test_set.targets)

    # Step 3: Define train/validation split size
    split = val_split

    # Step 4: Build transformation pipelines
    train_transform = _build_augmentation_transform(config)
    test_transform = build_normalization_transform()

    # Step 4.1: Verify augmentation pipeline using one sample
    sample_aug = train_transform(train_images[0])
    print("\n🎛️  Augmentation pipeline verification:")
    print(f"→ Shape: {sample_aug.shape}")
    print(f"→ Min: {sample_aug.min().item():.3f}, Max: {sample_aug.max().item():.3f}")

    # Step 5: Transform all training images
    train_data = [train_transform(img).permute(1, 2, 0).numpy() for img in train_images]

    # Step 6: Transform all test images
    test_data = [test_transform(img).permute(1, 2, 0).numpy() for img in test_images]

    # Step 7: Convert to arrays
    train_data = np.stack(train_data).astype(np.float32)
    test_data = np.stack(test_data).astype(np.float32)

    # Step 8: Split training set into train/val
    val_data = train_data[:split]
    val_labels = train_labels[:split]
    train_data_final = train_data[split:]
    train_labels_final = train_labels[split:]

    # Step 9: Return all arrays
    return train_data_final, train_labels_final, val_data, val_labels, test_data, test_labels


# Print module successfully executed
print("\n✅  data.py successfully executed.")
