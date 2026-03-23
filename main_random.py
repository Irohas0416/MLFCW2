"""
main_random.py
--------------
Random baseline active learning experiment on CIFAR-10.

Usage
-----
    python main_random.py --rounds 5 --budget 10 --reps 3 --clf_epochs 100
"""

import os
import argparse
import json
import numpy as np
import torch

from datasets             import get_cifar10, get_loaders
from extract_embeddings   import load_or_extract
from selection_strategies import RandomSelector
from train                import build_classifier, train_classifier, evaluate


def run_random(n_total:      int,
               train_dataset,
               test_loader,
               device:       torch.device,
               budget:       int = 10,
               n_rounds:     int = 5,
               n_reps:       int = 3,
               clf_epochs:   int = 100,
               seed:         int = 42) -> dict:
    """
    Random AL baseline loop. Mirrors run_tpcrp() exactly for fair comparison.
    """
    selector = RandomSelector(seed=seed)
    all_acc  = []

    for rep in range(n_reps):
        print(f"\n── Random  rep {rep+1}/{n_reps} ──")
        labeled  = set()
        rep_acc  = []
        cum_bgt  = 0

        for rnd in range(n_rounds):
            cum_bgt += budget
            print(f"\n  ┌─ Round {rnd+1}/{n_rounds}  |  Budget {cum_bgt}  "
                  f"(Random selecting…)", flush=True)

            new_idx = selector.select(budget, labeled, n_total)
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
    return {'budgets': budgets,
            'mean_acc': all_acc.mean(axis=0).tolist(),
            'std_acc':  all_acc.std(axis=0).tolist(),
            'all_acc':  all_acc.tolist()}



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir',    type=str, default='./data')
    parser.add_argument('--cache_dir',   type=str, default='./cache')
    parser.add_argument('--results_dir', type=str, default='./results')
    parser.add_argument('--clf_epochs',  type=int, default=100)
    parser.add_argument('--budget',      type=int, default=10)
    parser.add_argument('--rounds',      type=int, default=5)
    parser.add_argument('--reps',        type=int, default=3)
    parser.add_argument('--workers',     type=int, default=2)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    train_simclr_ds, train_eval_ds, test_ds = get_cifar10(args.data_dir)
    _, test_loader = get_loaders(train_simclr_ds, test_ds,
                                  num_workers=args.workers)

    rand_results = run_random(
        n_total      = len(train_eval_ds),
        train_dataset = train_eval_ds,
        test_loader  = test_loader,
        device       = device,
        budget       = args.budget,
        n_rounds     = args.rounds,
        n_reps       = args.reps,
        clf_epochs   = args.clf_epochs,
    )

    os.makedirs(args.results_dir, exist_ok=True)
    rand_path = os.path.join(args.results_dir, 'random_results.json')
    with open(rand_path, 'w') as f:
        json.dump({k: v for k, v in rand_results.items() if k != 'all_acc'}, f, indent=2)
    print(f"Random results saved → {rand_path}")
    print("Run compare.py to generate plots.")


if __name__ == '__main__':
    main()
