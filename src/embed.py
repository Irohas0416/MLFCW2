import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data import get_cifar10, get_eval_transform
from src.simclr import SimCLR

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, default="./data")
    ap.add_argument("--ckpt", type=str, default="checkpoints/simclr.pt")
    ap.add_argument("--split", type=str, choices=["train", "test"], default="train")
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--out_dir", type=str, default="embeddings")
    args = ap.parse_args()

    device = get_device()
    print("device:", device)

    model = SimCLR(proj_dim=128)
    ckpt = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=True)
    model.to(device)
    model.eval()

    transform = get_eval_transform()
    train_flag = (args.split == "train")
    ds = get_cifar10(
        root=args.data_root,
        train=train_flag,
        transform=transform,
        download=True,
        indexed=True
    )

    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
        persistent_workers=(args.num_workers > 0),
    )

    feats = []
    idxs = []
    for x, y, idx in tqdm(loader, desc="embed"):
        x = x.to(device, non_blocking=True)
        h = model.encode(x)
        feats.append(h.cpu().numpy())
        idxs.append(idx.numpy())

    feats = np.concatenate(feats, axis=0)
    idxs = np.concatenate(idxs, axis=0)

    os.makedirs(args.out_dir, exist_ok=True)
    feat_path = os.path.join(args.out_dir, f"cifar10_{args.split}_embeddings.npy")
    idx_path = os.path.join(args.out_dir, f"cifar10_{args.split}_indices.npy")

    np.save(feat_path, feats)
    np.save(idx_path, idxs)

    print("saved:", feat_path, feats.shape)
    print("saved:", idx_path, idxs.shape)
    print("first 10 indices:", idxs[:10])

if __name__ == "__main__":
    main()