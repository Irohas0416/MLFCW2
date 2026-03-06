import os
import json
import time
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from src.data import get_cifar10_subset
from src.classifier import build_resnet18_cifar10

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2470, 0.2435, 0.2616)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_train_transform():
    return transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])


def get_test_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])


def accuracy(logits, y):
    pred = logits.argmax(dim=1)
    return (pred == y).float().mean().item()


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    n_batches = 0

    for batch in tqdm(loader, desc="train", leave=False):
        x, y, idx = batch
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x)
        loss = criterion(logits, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_acc += accuracy(logits, y)
        n_batches += 1

    return total_loss / max(n_batches, 1), total_acc / max(n_batches, 1)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    n_batches = 0

    for batch in tqdm(loader, desc="test", leave=False):
        x, y, idx = batch
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x)
        loss = criterion(logits, y)

        total_loss += loss.item()
        total_acc += accuracy(logits, y)
        n_batches += 1

    return total_loss / max(n_batches, 1), total_acc / max(n_batches, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, default="./data")
    ap.add_argument("--selected_indices", type=str, required=True)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=0.025)
    ap.add_argument("--weight_decay", type=float, default=5e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--out_dir", type=str, default="results")
    ap.add_argument("--tag", type=str, default="tpcrp")
    args = ap.parse_args()

    set_seed(args.seed)
    device = get_device()
    print("device:", device)

    os.makedirs(args.out_dir, exist_ok=True)

    selected_indices = np.load(args.selected_indices)
    print("loaded selected indices:", selected_indices.shape)
    print("first indices:", selected_indices[:min(10, len(selected_indices))])

    train_ds = get_cifar10_subset(
        root=args.data_root,
        train=True,
        transform=get_train_transform(),
        download=True,
        indices=selected_indices,
        indexed=True
    )

    test_ds = get_cifar10_subset(
        root=args.data_root,
        train=False,
        transform=get_test_transform(),
        download=True,
        indices=None,
        indexed=True
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=min(args.batch_size, len(train_ds)),
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=256,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
    )

    model = build_resnet18_cifar10(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=0.9,
        nesterov=True,
        weight_decay=args.weight_decay
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs
    )

    best_test_acc = 0.0
    best_epoch = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        current_lr = optimizer.param_groups[0]["lr"]

        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        test_loss, test_acc = evaluate(
            model, test_loader, criterion, device
        )

        scheduler.step()

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_epoch = epoch

        row = {
            "epoch": epoch,
            "lr": current_lr,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "test_loss": test_loss,
            "test_acc": test_acc,
            "best_test_acc": best_test_acc,
            "best_epoch": best_epoch,
        }
        history.append(row)

        dt = time.time() - t0
        print(
            f"epoch {epoch:03d}/{args.epochs} "
            f"lr={current_lr:.6f} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"test_loss={test_loss:.4f} test_acc={test_acc:.4f} "
            f"best={best_test_acc:.4f} best_epoch={best_epoch} "
            f"time={dt:.1f}s"
        )

    budget = len(selected_indices)
    out_json = os.path.join(args.out_dir, f"{args.tag}_B{budget}_seed{args.seed}.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({
            "tag": args.tag,
            "budget": int(budget),
            "seed": int(args.seed),
            "selected_indices_path": args.selected_indices,
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "best_test_acc": float(best_test_acc),
            "best_epoch": int(best_epoch),
            "final_test_acc": float(history[-1]["test_acc"]),
            "history": history,
        }, f, indent=2)

    print("saved:", out_json)


if __name__ == "__main__":
    main()