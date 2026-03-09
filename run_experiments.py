import subprocess
import sys

SEEDS = [0, 1, 2]
BUDGETS = [10, 50]

EPOCHS = 100
BATCH_SIZE = 64
LR = 0.025
WEIGHT_DECAY = 5e-4
NUM_WORKERS = 0


def run(cmd):
    print("\n" + "=" * 100)
    print("RUN:", " ".join(cmd))
    print("=" * 100)
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {result.returncode}")


def main():
    py = sys.executable

    # Step 1: generate random selections for each seed/budget
    for seed in SEEDS:
        for budget in BUDGETS:
            run([
                py, "-m", "src.random_select",
                "--budget", str(budget),
                "--seed", str(seed),
                "--out_dir", "results",
            ])

    # Step 2: run supervised evaluation
    for seed in SEEDS:
        for budget in BUDGETS:
            # TPCRP
            run([
                py, "-m", "src.train_supervised_subset",
                "--selected_indices", f"results/tpcrp_selected_B{budget}.npy",
                "--epochs", str(EPOCHS),
                "--batch_size", str(BATCH_SIZE),
                "--lr", str(LR),
                "--weight_decay", str(WEIGHT_DECAY),
                "--seed", str(seed),
                "--num_workers", str(NUM_WORKERS),
                "--tag", "tpcrp",
            ])

            # Random
            run([
                py, "-m", "src.train_supervised_subset",
                "--selected_indices", f"results/random_selected_B{budget}_seed{seed}.npy",
                "--epochs", str(EPOCHS),
                "--batch_size", str(BATCH_SIZE),
                "--lr", str(LR),
                "--weight_decay", str(WEIGHT_DECAY),
                "--seed", str(seed),
                "--num_workers", str(NUM_WORKERS),
                "--tag", "random",
            ])

    print("\nAll experiments finished.")


if __name__ == "__main__":
    main()