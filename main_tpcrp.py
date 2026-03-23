"""
main_tpcrp.py
-------------
TPCRP active learning experiment on CIFAR-10.

Usage
-----
    python main_tpcrp.py --rounds 5 --budget 10 --reps 3 --clf_epochs 100
"""

import os
import argparse
import json
import numpy as np
import torch

from datasets             import get_cifar10, get_loaders
from pretrain_simclr      import SimCLRModel, train_simclr
from extract_embeddings   import load_or_extract
from selection_strategies import TPCRPSelector
from train                import build_classifier, train_classifier, evaluate


def run_tpcrp(embeddings:    np.ndarray,
              train_dataset,
              test_loader,
              device:        torch.device,
              budget:        int = 10,
              n_rounds:      int = 5,
              n_reps:        int = 3,
              clf_epochs:    int = 100,
              max_clusters:  int = 500,
              seed:          int = 42) -> dict:
    """
    Run the TPCRP active learning loop.

    Returns
    -------
    dict with keys:
        'budgets'     : list of cumulative budgets per round
        'mean_acc'    : mean accuracy over repetitions  (list)
        'std_acc'     : std  accuracy over repetitions  (list)
        'all_acc'     : raw  accuracy matrix (reps × rounds)
    """
    selector    = TPCRPSelector(max_clusters=max_clusters, seed=seed)
    all_acc     = []

    for rep in range(n_reps):
        print(f"\n── TPCRP  rep {rep+1}/{n_reps} ──")
        labeled  = set()
        rep_acc  = []
        cum_bgt  = 0

        for rnd in range(n_rounds):
            cum_bgt += budget
            print(f"\n  ┌─ Round {rnd+1}/{n_rounds}  |  Budget {cum_bgt}  "
                  f"({'TPCRP selecting…'})", flush=True)

            new_idx = selector.select(embeddings, budget,
                                      labeled_indices=labeled)
            labeled.update(new_idx)
            print(f"  │  Selected {len(new_idx)} samples  "
                  f"(total labeled: {len(labeled)})", flush=True)

            print(f"  │  Training classifier for {clf_epochs} epochs…", flush=True)
            clf = build_classifier().to(device)
            clf = train_classifier(clf, list(labeled), train_dataset,
                                   device, epochs=clf_epochs)

            acc = evaluate(clf, test_loader, device)
            rep_acc.append(acc)
            print(f"  └─ Acc: {acc:.2f}%", flush=True)

        all_acc.append(rep_acc)

    budgets  = [budget * (r+1) for r in range(n_rounds)]
    all_acc  = np.array(all_acc)
    mean_acc = all_acc.mean(axis=0).tolist()
    std_acc  = all_acc.std(axis=0).tolist()

    return {'budgets': budgets, 'mean_acc': mean_acc,
            'std_acc': std_acc, 'all_acc': all_acc.tolist()}



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir',       type=str, default='./data')
    parser.add_argument('--cache_dir',      type=str, default='./cache')
    parser.add_argument('--results_dir',    type=str, default='./results')
    parser.add_argument('--simclr_epochs',  type=int, default=30,
                        help='SimCLR pretraining epochs (paper=200, demo=30)')
    parser.add_argument('--clf_epochs',     type=int, default=100)
    parser.add_argument('--budget',         type=int, default=10,
                        help='Labeled examples queried per round (=num_classes)')
    parser.add_argument('--rounds',         type=int, default=5)
    parser.add_argument('--reps',           type=int, default=3)
    parser.add_argument('--max_clusters',   type=int, default=500)
    parser.add_argument('--workers',        type=int, default=2)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    train_simclr_ds, train_eval_ds, test_ds = get_cifar10(args.data_dir)
    simclr_loader, test_loader = get_loaders(
        train_simclr_ds, test_ds, num_workers=args.workers
    )

    simclr_path = os.path.join(args.cache_dir, 'simclr.pth')
    if not os.path.exists(simclr_path):
        print(f"\nNo cached SimCLR found. Training for {args.simclr_epochs} epochs …")
        model = SimCLRModel().to(device)
        train_simclr(model, simclr_loader,
                     epochs=args.simclr_epochs,
                     device=str(device), save_path=simclr_path)
    else:
        print(f"SimCLR checkpoint found: {simclr_path}")

    emb_path = os.path.join(args.cache_dir, 'embeddings.npz')
    embeddings, _ = load_or_extract(simclr_path, emb_path,
                                    data_dir=args.data_dir,
                                    workers=args.workers)
    print(f"Embeddings: {embeddings.shape}")

    results = run_tpcrp(embeddings, train_eval_ds, test_loader,
                        device       = device,
                        budget       = args.budget,
                        n_rounds     = args.rounds,
                        n_reps       = args.reps,
                        clf_epochs   = args.clf_epochs,
                        max_clusters = args.max_clusters)

    os.makedirs(args.results_dir, exist_ok=True)
    out_path = os.path.join(args.results_dir, 'tpcrp_results.json')
    with open(out_path, 'w') as f:
        json.dump({k: v for k, v in results.items() if k != 'all_acc'}, f, indent=2)
    print(f"\nResults saved → {out_path}")

    print("\n=== TPCRP Results ===")
    for b, m, s in zip(results['budgets'], results['mean_acc'], results['std_acc']):
        print(f"  Budget {b:>4d}: {m:.2f}% ± {s:.2f}%")


if __name__ == '__main__':
    main()
