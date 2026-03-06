import os
import argparse
import numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool_size", type=int, default=50000)
    ap.add_argument("--budget", type=int, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_dir", type=str, default="results")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    selected = rng.choice(args.pool_size, size=args.budget, replace=False)

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f"random_selected_B{args.budget}_seed{args.seed}.npy")
    np.save(out_path, selected)

    print("selected:", selected)
    print("saved:", out_path)

if __name__ == "__main__":
    main()