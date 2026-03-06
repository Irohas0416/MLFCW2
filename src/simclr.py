import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18

class SimCLR(nn.Module):
    """
    Backbone: ResNet-18
    Projection head: 2-layer MLP
    encode(x) returns representation before projection head.
    forward(x) returns projected features for contrastive loss.
    """
    def __init__(self, proj_dim=128):
        super().__init__()
        backbone = resnet18(weights=None)
        # modify first conv for CIFAR-10 (32x32): kernel 3, stride 1, remove maxpool
        backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        backbone.maxpool = nn.Identity()

        self.backbone = backbone
        self.feat_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()

        self.projector = nn.Sequential(
            nn.Linear(self.feat_dim, self.feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.feat_dim, proj_dim)
        )

    @torch.no_grad()
    def encode(self, x):
        return self.backbone(x)

    def forward(self, x):
        h = self.backbone(x)
        z = self.projector(h)
        z = F.normalize(z, dim=1)
        return z

def nt_xent_loss(z1, z2, temperature=0.2):
    """
    NT-Xent loss for SimCLR.
    z1, z2: (N, D), already normalized
    """
    N = z1.size(0)
    z = torch.cat([z1, z2], dim=0)  # (2N, D)

    # cosine similarity
    sim = torch.matmul(z, z.t()) / temperature  # (2N, 2N)

    # mask self-similarity
    mask = torch.eye(2*N, device=z.device, dtype=torch.bool)
    sim = sim.masked_fill(mask, float("-inf"))

    # positives: i <-> i+N
    pos = torch.cat([torch.arange(N, 2*N), torch.arange(0, N)]).to(z.device)
    loss = F.cross_entropy(sim, pos)
    return loss