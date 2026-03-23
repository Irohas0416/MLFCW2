"""
pretrain_simclr.py
------------------
Step 1 of TPCRP: Self-supervised representation learning with SimCLR.

Architecture  : ResNet-18 backbone + MLP projection head (128-d output)
Loss          : NT-Xent contrastive loss (temperature = 0.5)
Optimiser     : SGD, cosine LR schedule
Reference     : Chen et al. (2020), App. F.1 of Hacohen et al. (2022)

Usage
-----
    python pretrain_simclr.py --epochs 200 --save cache/simclr.pth
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models

from datasets import get_cifar10, get_loaders


class ProjectionHead(nn.Module):
    """Two-layer MLP projection head (as in SimCLR / paper App. F.1)."""
    def __init__(self, in_dim: int = 512, hidden_dim: int = 256, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SimCLRModel(nn.Module):
    """
    ResNet-18 backbone with MLP projection head.
    Forward returns (h, z):
      h : penultimate 512-d features  (used for TPCRP queries)
      z : 128-d projected features    (used for contrastive loss)
    """
    def __init__(self):
        super().__init__()
        resnet = models.resnet18(weights=None)
        self.backbone   = nn.Sequential(*list(resnet.children())[:-1])
        self.projection = ProjectionHead(512, 256, 128)

    def forward(self, x: torch.Tensor):
        h = self.backbone(x).flatten(start_dim=1)
        z = self.projection(h)
        return h, z


def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor,
                 temperature: float = 0.5) -> torch.Tensor:
    """
    NT-Xent contrastive loss.
    Positive pairs: (z1[i], z2[i]) for each i in the batch.
    """
    B = z1.size(0)
    z = torch.cat([z1, z2], dim=0)                  # (2B, D)
    z = nn.functional.normalize(z, dim=1)
    sim = torch.mm(z, z.T) / temperature             # (2B, 2B)

    # Mask diagonal (self-similarity → −∞ so softmax ignores it)
    mask = torch.eye(2 * B, dtype=torch.bool, device=z.device)
    sim.masked_fill_(mask, float('-inf'))

    # Positive pair for sample i is at index i+B (and vice-versa)
    labels = torch.cat([torch.arange(B, 2*B), torch.arange(B)]).to(z.device)
    return nn.functional.cross_entropy(sim, labels)


def train_simclr(model: SimCLRModel,
                 loader,
                 epochs:      int   = 200,
                 lr:          float = 0.4,
                 weight_decay:float = 1e-4,
                 temperature: float = 0.5,
                 device:      str   = 'cpu',
                 save_path:   str   = 'cache/simclr.pth') -> list:
    """
    Train SimCLR and save checkpoint.
    Returns list of per-epoch average losses.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    optimizer = optim.SGD(model.parameters(), lr=lr,
                          momentum=0.9, weight_decay=weight_decay)
    # 10-epoch linear warmup → cosine annealing (standard SimCLR schedule)
    warmup_epochs = min(10, epochs)
    warmup = optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs)
    cosine = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, epochs - warmup_epochs))
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs])

    model.train()
    loss_history = []

    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        for (x1, x2), _ in loader:
            x1, x2 = x1.to(device), x2.to(device)
            _, z1 = model(x1)
            _, z2 = model(x2)
            loss = nt_xent_loss(z1, z2, temperature)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()
        avg = epoch_loss / len(loader)
        loss_history.append(avg)

        if epoch % 10 == 0 or epoch == 1:
            print(f"  [SimCLR] Epoch {epoch:>3d}/{epochs}  Loss: {avg:.4f}")

    torch.save(model.state_dict(), save_path)
    print(f"\nSimCLR checkpoint saved → {save_path}")
    return loss_history


def main():
    parser = argparse.ArgumentParser(description='Pretrain SimCLR on CIFAR-10')
    parser.add_argument('--epochs',     type=int,   default=200)
    parser.add_argument('--lr',         type=float, default=0.3)
    parser.add_argument('--batch_size', type=int,   default=256)
    parser.add_argument('--data_dir',   type=str,   default='./data')
    parser.add_argument('--save',       type=str,   default='cache/simclr.pth')
    parser.add_argument('--workers',    type=int,   default=2)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Training SimCLR for {args.epochs} epochs …")

    train_simclr_ds, _, _ = get_cifar10(args.data_dir)
    simclr_loader, _      = get_loaders(train_simclr_ds, None,
                                         simclr_batch=args.batch_size,
                                         num_workers=args.workers)

    model = SimCLRModel().to(device)
    train_simclr(model, simclr_loader,
                 epochs=args.epochs, lr=args.lr,
                 device=str(device), save_path=args.save)


if __name__ == '__main__':
    main()
