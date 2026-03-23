"""
compare.py
----------
Load saved results and produce comparison plots.

Automatically detects which result files are present:
  - tpcrp only          → single-curve plot
  - tpcrp + random      → two-way comparison
  - all three           → three-way comparison

Usage
-----
    python compare.py --results_dir ./results
"""

import os
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt


def plot_single(tpcrp_res: dict, save_dir: str):
    budgets  = tpcrp_res['budgets']
    mean_acc = np.array(tpcrp_res['mean_acc'])
    std_acc  = np.array(tpcrp_res['std_acc'])

    plt.figure(figsize=(8, 5))
    plt.plot(budgets, mean_acc, 'o-', color='royalblue', label='TPCRP')
    plt.fill_between(budgets, mean_acc - std_acc, mean_acc + std_acc,
                     alpha=0.2, color='royalblue')
    plt.xlabel('Cumulative labeled examples (budget)')
    plt.ylabel('Test accuracy (%)')
    plt.title('TPCRP – CIFAR-10 (Fully Supervised)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(save_dir, 'tpcrp_accuracy.png')
    plt.savefig(path, dpi=150)
    print(f"Plot saved → {path}")
    plt.close()


def plot_comparison(tpcrp_res: dict, rand_res: dict, save_dir: str):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    def _plot(ax, res, label, color):
        b = res['budgets']
        m = np.array(res['mean_acc'])
        s = np.array(res['std_acc'])
        ax.plot(b, m, 'o-', label=label, color=color)
        ax.fill_between(b, m - s, m + s, alpha=0.15, color=color)

    _plot(axes[0], tpcrp_res, 'TPCRP (original)', 'royalblue')
    _plot(axes[0], rand_res,  'Random',            'tomato')
    axes[0].set_xlabel('Cumulative labeled examples')
    axes[0].set_ylabel('Test accuracy (%)')
    axes[0].set_title('Active Learning on CIFAR-10 – Low Budget Regime')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    budgets = tpcrp_res['budgets']
    diff    = np.array(tpcrp_res['mean_acc']) - np.array(rand_res['mean_acc'])
    axes[1].bar(budgets, diff, color='royalblue', alpha=0.7, label='TPCRP – Random')
    axes[1].axhline(0, color='black', linewidth=0.8)
    axes[1].set_xlabel('Cumulative labeled examples')
    axes[1].set_ylabel('Accuracy gain over Random (%)')
    axes[1].set_title('Performance Gain vs. Random Baseline')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, 'comparison.png')
    plt.savefig(path, dpi=150)
    print(f"Comparison plot saved → {path}")
    plt.close()

    print("\n=== Results Table ===")
    header = f"{'Budget':>8}  {'TPCRP (orig)':>16}  {'Random':>16}  {'Gain':>8}"
    print(header)
    print("-" * len(header))
    for b, tm, ts, rm, rs in zip(budgets,
                                  tpcrp_res['mean_acc'], tpcrp_res['std_acc'],
                                  rand_res['mean_acc'],  rand_res['std_acc']):
        print(f"{b:>8d}  {tm:>6.2f}±{ts:<6.2f}    {rm:>6.2f}±{rs:<6.2f}    {tm-rm:>+.2f}%")


def plot_three_way(orig_res: dict, mod_res: dict, rand_res: dict, save_dir: str):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for res, label, color in [
        (orig_res, 'TPCRP original',   'royalblue'),
        (mod_res,  'TPCRP Adaptive-K', 'seagreen'),
        (rand_res, 'Random',           'tomato'),
    ]:
        b = np.array(res['budgets'])
        m = np.array(res['mean_acc'])
        s = np.array(res['std_acc'])
        axes[0].plot(b, m, 'o-', label=label, color=color)
        axes[0].fill_between(b, m - s, m + s, alpha=0.15, color=color)

    axes[0].set_xlabel('Cumulative labeled examples')
    axes[0].set_ylabel('Test accuracy (%)')
    axes[0].set_title('TPCRP Original vs Adaptive-K vs Random (CIFAR-10)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    budgets    = np.array(orig_res['budgets'])
    diff       = np.array(mod_res['mean_acc']) - np.array(orig_res['mean_acc'])
    bar_colors = ['seagreen' if d >= 0 else 'tomato' for d in diff]
    axes[1].bar(budgets, diff, color=bar_colors, alpha=0.8)
    axes[1].axhline(0, color='black', linewidth=0.8)
    for b, d in zip(budgets, diff):
        axes[1].text(b, d + 0.05 * np.sign(d), f'{d:+.1f}%',
                     ha='center', va='bottom' if d >= 0 else 'top', fontsize=9)
    axes[1].set_xlabel('Cumulative labeled examples')
    axes[1].set_ylabel('Accuracy improvement: Adaptive-K – Original (%)')
    axes[1].set_title('Gain from Adaptive-K Modification')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, 'modification_comparison.png')
    plt.savefig(path, dpi=150)
    print(f"Plot saved → {path}")
    plt.close()

    print("\n=== Three-way Results Table ===")
    hdr = (f"{'Budget':>8}  {'Original':>14}  "
           f"{'Adaptive-K':>14}  {'Random':>14}  {'Gain vs Orig':>14}")
    print(hdr)
    print("─" * len(hdr))
    for b, om, os_, mm, ms, rm, rs in zip(
            orig_res['budgets'],
            orig_res['mean_acc'], orig_res['std_acc'],
            mod_res['mean_acc'],  mod_res['std_acc'],
            rand_res['mean_acc'], rand_res['std_acc']):
        print(f"{b:>8d}  {om:>5.2f}±{os_:<5.2f}    "
              f"{mm:>5.2f}±{ms:<5.2f}    "
              f"{rm:>5.2f}±{rs:<5.2f}    {mm-om:>+.2f}%")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', type=str, default='./results')
    args = parser.parse_args()

    d = args.results_dir
    tpcrp_path = os.path.join(d, 'tpcrp_results.json')
    rand_path  = os.path.join(d, 'random_results.json')
    mod_path   = os.path.join(d, 'modified_results.json')

    if not os.path.exists(tpcrp_path):
        print("tpcrp_results.json not found. Run main_tpcrp.py first.")
        return

    with open(tpcrp_path) as f:
        tpcrp_res = json.load(f)

    os.makedirs(d, exist_ok=True)

    if not os.path.exists(rand_path):
        print("random_results.json not found — plotting TPCRP only.")
        plot_single(tpcrp_res, d)
        return

    with open(rand_path) as f:
        rand_res = json.load(f)

    if os.path.exists(mod_path):
        with open(mod_path) as f:
            mod_res = json.load(f)
        plot_three_way(tpcrp_res, mod_res, rand_res, d)
    else:
        plot_comparison(tpcrp_res, rand_res, d)


if __name__ == '__main__':
    main()
