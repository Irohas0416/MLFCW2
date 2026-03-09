import os
import json
import csv
import math
from glob import glob

RESULTS_DIR = "results"


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def std(xs):
    if len(xs) <= 1:
        return 0.0
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def parse_name(path):
    """
    filename like:
      tpcrp_B10_seed0.json
      random_B50_seed2.json
    """
    name = os.path.basename(path).replace(".json", "")
    parts = name.split("_")
    method = parts[0]
    budget = int(parts[1].replace("B", ""))
    seed = int(parts[2].replace("seed", ""))
    return method, budget, seed


def main():
    files = sorted(glob(os.path.join(RESULTS_DIR, "*.json")))
    if not files:
        print("No json files found in results/")
        return

    rows = []
    grouped = {}

    for path in files:
        method, budget, seed = parse_name(path)

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        row = {
            "method": method,
            "budget": budget,
            "seed": seed,
            "best_test_acc": float(data["best_test_acc"]),
            "best_epoch": int(data["best_epoch"]),
            "final_test_acc": float(data["final_test_acc"]),
            "path": path,
        }
        rows.append(row)

        key = (method, budget)
        grouped.setdefault(key, []).append(row)

    # write detailed csv
    detail_csv = os.path.join(RESULTS_DIR, "summary_detail.csv")
    with open(detail_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "method",
                "budget",
                "seed",
                "best_test_acc",
                "best_epoch",
                "final_test_acc",
                "path",
            ],
        )
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda x: (x["budget"], x["method"], x["seed"])))

    # write aggregated csv
    agg_rows = []
    for (method, budget), items in sorted(grouped.items(), key=lambda x: (x[0][1], x[0][0])):
        bests = [r["best_test_acc"] for r in items]
        finals = [r["final_test_acc"] for r in items]
        best_epochs = [r["best_epoch"] for r in items]

        agg_rows.append({
            "method": method,
            "budget": budget,
            "n_seeds": len(items),
            "best_test_acc_mean": mean(bests),
            "best_test_acc_std": std(bests),
            "final_test_acc_mean": mean(finals),
            "final_test_acc_std": std(finals),
            "best_epoch_mean": mean(best_epochs),
        })

    agg_csv = os.path.join(RESULTS_DIR, "summary_aggregate.csv")
    with open(agg_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "method",
                "budget",
                "n_seeds",
                "best_test_acc_mean",
                "best_test_acc_std",
                "final_test_acc_mean",
                "final_test_acc_std",
                "best_epoch_mean",
            ],
        )
        writer.writeheader()
        writer.writerows(agg_rows)

    # print summary to terminal
    print("\nDetailed results saved to:", detail_csv)
    print("Aggregated results saved to:", agg_csv)
    print("\n=== Mean ± Std of best_test_acc ===")
    for row in agg_rows:
        print(
            f"{row['method']:>6}  B={row['budget']:>2}  "
            f"{row['best_test_acc_mean']:.4f} ± {row['best_test_acc_std']:.4f}  "
            f"(n={row['n_seeds']})"
        )


if __name__ == "__main__":
    main()