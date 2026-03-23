"""
datasets.py
-----------
CIFAR-10 data loading utilities.
Provides:
  - SimCLR double-augmentation transform (for self-supervised pretraining)
  - Standard eval transform (for classifier training / testing)
  - Helper to get train/test DataLoaders
"""

import warnings
import numpy as np

# Suppress NumPy 2.4 / torchvision pickle compatibility warning.
# torchvision's CIFAR loader passes old-style dtype kwargs to pickle;
# this is a known upstream issue and harmless — suppress until patched.
warnings.filterwarnings(
    "ignore",
    message=r".*dtype\(\).*align.*",
    category=np.exceptions.VisibleDeprecationWarning,
)

import torch
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms

# ── CIFAR-10 normalisation constants ──────────────────────────────────────────
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2023, 0.1994, 0.2010)

CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']

# ── Transforms ────────────────────────────────────────────────────────────────

class SimCLRTransform:
    """
    Returns TWO independently augmented views of the same image.
    Used during SimCLR self-supervised pretraining (Step 1 of TPCRP).
    Follows the augmentation policy described in Chen et al. (2020) / paper App. F.1.
    """
    def __init__(self, img_size: int = 32):
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(size=img_size, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                   saturation=0.4, hue=0.1),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ])

    def __call__(self, img):
        return self.transform(img), self.transform(img)


EVAL_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
])

# ── Dataset helpers ───────────────────────────────────────────────────────────

def get_cifar10(data_dir: str = './data'):
    """
    Download (if needed) and return CIFAR-10 datasets.

    Returns
    -------
    train_simclr : dataset with SimCLRTransform  (for pretraining)
    train_eval   : dataset with EVAL_TRANSFORM   (for AL classifier training)
    test_eval    : dataset with EVAL_TRANSFORM   (for accuracy measurement)
    """
    train_simclr = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=SimCLRTransform()
    )
    train_eval = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=EVAL_TRANSFORM
    )
    test_eval = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=EVAL_TRANSFORM
    )
    return train_simclr, train_eval, test_eval


def get_loaders(train_simclr, test_eval,
                simclr_batch: int = 256,
                test_batch:   int = 512,
                num_workers:  int = 2):
    """Return DataLoaders for SimCLR pretraining and test evaluation."""
    simclr_loader = DataLoader(
        train_simclr, batch_size=simclr_batch,
        shuffle=True, num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_eval, batch_size=test_batch,
        shuffle=False, num_workers=num_workers, pin_memory=True
    )
    return simclr_loader, test_loader


def get_labeled_loader(train_eval, labeled_indices: list,
                       batch_size: int = 64, num_workers: int = 2):
    """Return a DataLoader over a labeled subset (for classifier training)."""
    subset = Subset(train_eval, labeled_indices)
    return DataLoader(subset,
                      batch_size=min(batch_size, len(labeled_indices)),
                      shuffle=True,
                      num_workers=num_workers,
                      pin_memory=True)
