"""
train.py
--------
Classifier training and evaluation utilities.

Follows the "Fully Supervised" evaluation framework of
Munjal et al. (2020), as used in Hacohen et al. (2022):
  - ResNet-18 classifier
  - SGD + Nesterov + cosine LR schedule
  - Weights re-initialised between AL iterations
  - Augmentations: random crops + horizontal flips

Optimisations for small labeled sets (10-100 examples):
  - Entire labeled set cached as tensors on GPU (eliminates repeated I/O)
  - On-GPU augmentation via kornia (fast) or CPU pre-augmentation (fallback)
  - num_workers=0 (Windows multiprocessing workaround)
"""

import warnings
import numpy as np
warnings.filterwarnings(
    "ignore",
    message=r".*dtype\(\).*align.*",
    category=np.exceptions.VisibleDeprecationWarning,
)

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.datasets as tvd
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader, Subset
import random

from datasets import CIFAR10_MEAN, CIFAR10_STD

# ── Transforms ────────────────────────────────────────────────────────────────
TRAIN_TRANSFORM = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
])

EVAL_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
])

# ── GPU-side augmentation (applied each epoch on cached tensors) ───────────────

def gpu_augment(x: torch.Tensor) -> torch.Tensor:
    """
    Fast on-GPU augmentation for CIFAR-32x32 images.
    Equivalent to RandomCrop(32, padding=4) + RandomHorizontalFlip().
    Input: (N, C, H, W) float tensor already on GPU.
    Augmentation applied INDEPENDENTLY per image.
    """
    N = x.shape[0]
    # Pad each image by 4 on each side
    pad = 4
    x = torch.nn.functional.pad(x, (pad, pad, pad, pad), mode='reflect')
    # (N, C, 40, 40) now

    out = torch.empty(N, x.shape[1], 32, 32, device=x.device, dtype=x.dtype)
    for i in range(N):
        # Independent random crop per image
        top  = random.randint(0, pad * 2)   # 0..8
        left = random.randint(0, pad * 2)   # 0..8
        out[i] = x[i, :, top:top+32, left:left+32]
        # Independent random horizontal flip per image
        if random.random() > 0.5:
            out[i] = torch.flip(out[i], dims=[-1])
    return out


def build_classifier(num_classes: int = 10) -> nn.Module:
    """
    ResNet-18 with fresh random initialisation.
    Re-initialised between AL iterations.
    """
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(512, num_classes)
    return model


def train_classifier(model:           nn.Module,
                     labeled_indices: list,
                     train_dataset,
                     device:          torch.device,
                     epochs:          int   = 100,
                     lr:              float = 0.025,
                     weight_decay:    float = 5e-4,
                     batch_size:      int   = 64,
                     num_workers:     int   = 0) -> nn.Module:
    """
    Train classifier on labeled subset.
    Key optimisation: load all labeled images once, cache on GPU,
    apply augmentation each epoch directly on GPU tensors.
    This eliminates repeated disk I/O and CPU→GPU transfers.
    """
    # ── Load entire labeled set into GPU memory once ───────────────────────────
    # Use eval (non-augmented) transform to load raw normalised tensors,
    # then apply augmentation on GPU each epoch.
    base_ds = tvd.CIFAR10(root='./data', train=True, download=False,
                           transform=EVAL_TRANSFORM)
    subset  = Subset(base_ds, labeled_indices)
    loader  = DataLoader(subset, batch_size=len(labeled_indices),
                         shuffle=False, num_workers=0, pin_memory=False)

    # Cache all images + labels on GPU
    all_x, all_y = next(iter(loader))
    all_x = all_x.to(device)   # (N, 3, 32, 32)
    all_y = all_y.to(device)   # (N,)
    N = len(all_y)

    model     = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr,
                          momentum=0.9, weight_decay=weight_decay,
                          nesterov=True)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Always use full batch - eliminates small remainder batch edge cases
    effective_batch = N

    model.train()
    for epoch in range(epochs):
        # Shuffle indices
        perm    = torch.randperm(N, device=device)
        x_shuf  = all_x[perm]
        y_shuf  = all_y[perm]

        epoch_loss = 0.0
        for start in range(0, N, effective_batch):
            xb = x_shuf[start:start + effective_batch]
            yb = y_shuf[start:start + effective_batch]

            # Apply augmentation on GPU
            xb = gpu_augment(xb)

            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()

        if (epoch + 1) % 10 == 0 or epoch == 0:
            steps = max(1, N // effective_batch)
            print(f"      Epoch [{epoch+1:>3d}/{epochs}]  "
                  f"loss: {epoch_loss/steps:.4f}", flush=True)

    return model


@torch.no_grad()
def evaluate(model:       nn.Module,
             test_loader: DataLoader,
             device:      torch.device) -> float:
    """Top-1 test accuracy (%)."""
    model.eval()
    correct = total = 0
    for x, y in test_loader:
        x, y    = x.to(device), y.to(device)
        preds   = model(x).argmax(dim=1)
        correct += (preds == y).sum().item()
        total   += y.size(0)
    return 100.0 * correct / total