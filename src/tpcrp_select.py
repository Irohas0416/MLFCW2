import os
import argparse
import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.neighbors import NearestNeighbors

def l2_normalize(x, eps=1e-12):
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(norms, eps, None)

def compute_typicality(cluster_feats, k=20):
    """
    cluster_feats: (n_c, d)
    Typicality(x) = 1 / mean distance to K nearest neighbors
    Uses min(k, n_c) neighbors as in paper's practical handling.
    """
    n_c = len(cluster_feats)
    if n_c == 1:
        return np.array([1e12], dtype=np.float32)

    # include self in neighbors, then remove self distance later
    k_eff = min(k + 1, n_c)

    nn = NearestNeighbors(n_neighbors=k_eff, metric="euclidean")
    nn.fit(cluster_feats)
    dists, _ = nn.kneighbors(cluster_feats)   # (n_c, k_eff)

    # first neighbor is usually itself with distance 0
    if k_eff > 1:
        dists = dists[:, 1:]
    else:
        dists = dists

    mean_dist = dists.mean(axis=1)
    typicality = 1.0 / np.clip(mean_dist, 1e-12, None)
    return typicality.astype(np.float32)

def cluster_embeddings(x, n_clusters, seed=0):
    """
    Paper appendix: use KMeans when K <= 50, MiniBatchKMeans otherwise.
    """
    if n_clusters <= 50:
        model = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    else:
        model = MiniBatchKMeans(
            n_clusters=n_clusters,
            random_state=seed,
            batch_size=4096,
            n_init=10
        )
    labels = model.fit_predict(x)
    return labels

def select_tpcrp_initial_pool(features, orig_indices, budget, k_nn=20, seed=0):
    """
    Initial pool selection version of TPCRP / TypiClust:
    1) cluster into B clusters
    2) from each cluster choose the most typical sample
    """
    x = l2_normalize(features)

    cluster_labels = cluster_embeddings(x, n_clusters=budget, seed=seed)

    selected_dataset_indices = []
    debug_rows = []

    for cid in range(budget):
        member_mask = (cluster_labels == cid)
        member_pos = np.where(member_mask)[0]

        if len(member_pos) == 0:
            # theoretically rare; skip if empty
            continue

        cluster_feats = x[member_pos]
        typicality = compute_typicality(cluster_feats, k=k_nn)

        best_local = np.argmax(typicality)
        best_global_pos = member_pos[best_local]
        best_dataset_idx = orig_indices[best_global_pos]

        selected_dataset_indices.append(best_dataset_idx)

        debug_rows.append({
            "cluster_id": cid,
            "cluster_size": len(member_pos),
            "selected_pos_in_embedding": int(best_global_pos),
            "selected_dataset_index": int(best_dataset_idx),
            "selected_typicality": float(typicality[best_local]),
        })

    selected_dataset_indices = np.array(selected_dataset_indices, dtype=np.int64)

    return selected_dataset_indices, debug_rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embeddings", type=str, required=True)
    ap.add_argument("--indices", type=str, required=True)
    ap.add_argument("--budget", type=int, required=True)
    ap.add_argument("--k_nn", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_dir", type=str, default="results")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    feats = np.load(args.embeddings)
    idxs = np.load(args.indices)

    print("loaded embeddings:", feats.shape)
    print("loaded indices:", idxs.shape)
    print("budget:", args.budget)

    selected, debug_rows = select_tpcrp_initial_pool(
        features=feats,
        orig_indices=idxs,
        budget=args.budget,
        k_nn=args.k_nn,
        seed=args.seed,
    )

    out_selected = os.path.join(args.out_dir, f"tpcrp_selected_B{args.budget}.npy")
    np.save(out_selected, selected)

    print("selected count:", len(selected))
    print("selected indices:", selected[:min(20, len(selected))])
    print("saved:", out_selected)

    # optional debug csv
    import csv
    out_csv = os.path.join(args.out_dir, f"tpcrp_debug_B{args.budget}.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "cluster_id",
                "cluster_size",
                "selected_pos_in_embedding",
                "selected_dataset_index",
                "selected_typicality",
            ],
        )
        writer.writeheader()
        writer.writerows(debug_rows)

    print("saved:", out_csv)

if __name__ == "__main__":
    main()