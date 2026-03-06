import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from torch.utils.data import Subset

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2470, 0.2435, 0.2616)

class IndexedDataset(Dataset):
    """
    Wrap a dataset so __getitem__ returns (x, y, idx).
    idx must be stable so later TPCRP can select samples by index.
    """
    def __init__(self, base_ds):
        self.base = base_ds

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        x, y = self.base[idx]
        return x, y, idx

def get_simclr_transform(image_size=32):
    # Minimal SimCLR augmentations for CIFAR-10
    # (You can strengthen later; keep minimal first to reduce bugs)
    return transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

def get_cifar10_subset(root="./data", train=True, transform=None, download=True, indices=None, indexed=True):
    ds = get_cifar10(root=root, train=train, transform=transform, download=download, indexed=indexed)
    if indices is not None:
        ds = Subset(ds, indices.tolist() if hasattr(indices, "tolist") else list(indices))
    return ds

def get_eval_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

def get_cifar10(root="./data", train=True, transform=None, download=True, indexed=True):
    ds = datasets.CIFAR10(root=root, train=train, transform=transform, download=download)
    return IndexedDataset(ds) if indexed else ds

class TwoCropsTransform:
    """
    For SimCLR: returns two augmented views of same image.
    """
    def __init__(self, base_transform):
        self.t = base_transform

    def __call__(self, x):
        return self.t(x), self.t(x)