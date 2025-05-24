# Import third-party libraries
import random
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


# Function to load, optionally augment, and always standardize CIFAR-10
def build_dataset(config):
    """
    Loads the CIFAR-10 dataset, applies optional light mode slicing,
    optional data augmentation (crop, flip, cutout), and always performs
    per-channel normalization.

    Args:
        config (Config): Configuration object with DATA_PATH, LIGHT_MODE, AUGMENT_MODE

    Returns:
        tuple: (train_data, train_labels, test_data, test_labels)
               train/test data are np.float32 arrays with shape (N, 32, 32, 3)
    """

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

    # Define transformation pipeline
    def build_transform(augment):
        ops = [transforms.ToPILImage()]
        if augment:
            ops += [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                Cutout(size=16)  # Add Cutout during training only
            ]
        ops += [
            transforms.ToTensor(),
            transforms.Normalize(mean=_CIFAR10_MEAN, std=_CIFAR10_STD)
        ]
        return transforms.Compose(ops)

    # Build transforms for training and test
    train_transform = build_transform(config.AUGMENT_MODE)
    test_transform = build_transform(False)

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
