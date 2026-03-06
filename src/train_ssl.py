import os
import argparse
import time
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data import get_cifar10, get_simclr_transform, TwoCropsTransform
from src.simclr import SimCLR, nt_xent_loss

def set_seed(seed: int):
    import random, numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

def train_one_epoch(model, loader, opt, device, temperature):
    model.train()
    total_loss = 0.0
    n = 0
    for (x1, x2), _, _ in tqdm(loader, desc="train", leave=False):
        x1 = x1.to(device, non_blocking=True)
        x2 = x2.to(device, non_blocking=True)

        z1 = model(x1)
        z2 = model(x2)
        loss = nt_xent_loss(z1, z2, temperature=temperature)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        bs = x1.size(0)
        total_loss += loss.item() * bs
        n += bs
    return total_loss / max(n, 1)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, default="./data")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--out", type=str, default="checkpoints/simclr.pt")
    args = ap.parse_args()

    set_seed(args.seed)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    def get_device():
        if torch.cuda.is_available():
            return torch.device("cuda")
        # MPS for Apple Silicon (if available)
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    device = get_device()
    print("device:", device)

    print("device:", device)

    base_t = get_simclr_transform()
    ds = get_cifar10(root=args.data_root, train=True, transform=TwoCropsTransform(base_t), download=True, indexed=True)

    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
        persistent_workers=(args.num_workers > 0),
   )

    model = SimCLR(proj_dim=128).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        loss = train_one_epoch(model, loader, opt, device, args.temperature)
        dt = time.time() - t0
        print(f"epoch {epoch:03d}/{args.epochs}  loss={loss:.4f}  time={dt:.1f}s")

    ckpt = {
        "model": model.state_dict(),
        "args": vars(args),
    }
    torch.save(ckpt, args.out)
    print("saved:", args.out)

if __name__ == "__main__":
    main()