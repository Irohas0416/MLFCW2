"""
extract_embeddings.py
---------------------
Step 1 (post-train): Extract L2-normalised penultimate-layer embeddings
from the pretrained SimCLR model for the entire CIFAR-10 training set.
Caches the result to disk (numpy .npz) to avoid re-extraction every run.

Usage
-----
    python extract_embeddings.py --model cache/simclr.pth \
                                 --out   cache/embeddings.npz
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from datasets import get_cifar10, EVAL_TRANSFORM
from pretrain_simclr import SimCLRModel


def extract_embeddings(model:   SimCLRModel,
                       dataset,
                       device:  torch.device,
                       batch:   int = 512,
                       workers: int = 2) -> tuple:
    """
    Forward-pass the entire dataset through the backbone.

    Returns
    -------
    embeddings : np.ndarray  (N, 512)  L2-normalised penultimate features
    labels     : np.ndarray  (N,)      ground-truth class indices
    """
    loader = DataLoader(dataset, batch_size=batch,
                        shuffle=False, num_workers=workers, pin_memory=True)
    model.eval()
    embs, labs = [], []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            h, _ = model(x)                          # (B, 512)
            h = nn.functional.normalize(h, dim=1)    # L2 norm (paper App. F.1)
            embs.append(h.cpu().numpy())
            labs.append(y.numpy())

    return np.concatenate(embs, axis=0), np.concatenate(labs, axis=0)


def load_or_extract(model_path:  str,
                    out_path:    str,
                    data_dir:    str = './data',
                    device_str:  str = 'auto',
                    batch:       int = 512,
                    workers:     int = 2) -> tuple:
    """
    Load cached embeddings if they exist, otherwise extract and cache.

    Returns
    -------
    embeddings : np.ndarray  (N, 512)
    labels     : np.ndarray  (N,)
    """
    if os.path.exists(out_path):
        print(f"Loading cached embeddings from {out_path}")
        data = np.load(out_path)
        return data['embeddings'], data['labels']

    device = (torch.device('cuda' if torch.cuda.is_available() else 'cpu')
              if device_str == 'auto' else torch.device(device_str))

    print(f"Extracting embeddings with device={device} …")
    model = SimCLRModel().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    _, train_eval, _ = get_cifar10(data_dir)
    embeddings, labels = extract_embeddings(model, train_eval, device, batch, workers)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez(out_path, embeddings=embeddings, labels=labels)
    print(f"Embeddings saved → {out_path}  shape={embeddings.shape}")
    return embeddings, labels


def main():
    parser = argparse.ArgumentParser(description='Extract SimCLR embeddings for CIFAR-10')
    parser.add_argument('--model',    type=str, default='cache/simclr.pth')
    parser.add_argument('--out',      type=str, default='cache/embeddings.npz')
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--batch',    type=int, default=512)
    parser.add_argument('--workers',  type=int, default=2)
    args = parser.parse_args()

    load_or_extract(args.model, args.out,
                    data_dir=args.data_dir,
                    batch=args.batch, workers=args.workers)


if __name__ == '__main__':
    main()
