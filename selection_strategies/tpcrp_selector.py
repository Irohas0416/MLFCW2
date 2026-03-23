"""
selection_strategies/tpcrp_selector.py
---------------------------------------
TPCRP (TypiClust with Representation learning + K-means clustering).
Implements Algorithm 1 from Hacohen et al. (2022).

Also contains TPCRPModifiedSelector — the proposed modification
(Adaptive-K Typicality) for Task 3 of the coursework.

Key steps (Algorithm 1)
-----------------------
1. Representation learning  → done externally (pretrain_simclr.py)
2. Cluster embeddings into |L| + B clusters (K-means)
3. For each of the B largest *uncovered* clusters, pick the most
   typical point: highest Typicality = lowest mean KNN distance.
"""

import numpy as np
from sklearn.cluster    import KMeans, MiniBatchKMeans
from sklearn.neighbors  import NearestNeighbors
import random


# ── Typicality helper ─────────────────────────────────────────────────────────

def _typicality_fixed_k(embeddings: np.ndarray, K: int = 20) -> np.ndarray:
    """
    Equation (4) from the paper:
        Typicality(x) = (1/K * Σ ||x - x_i||)⁻¹   for x_i in K-NN(x)

    Parameters
    ----------
    embeddings : (N, D) array
    K          : number of nearest neighbours (paper default = 20)

    Returns
    -------
    typicality : (N,) array — higher means more typical (denser region)
    """
    N = len(embeddings)
    K_actual = min(K, N - 1)
    if K_actual < 1:
        return np.ones(N)

    nbrs = NearestNeighbors(n_neighbors=K_actual + 1,
                            metric='euclidean', n_jobs=-1).fit(embeddings)
    distances, _ = nbrs.kneighbors(embeddings)
    # Column 0 is self (distance = 0); skip it
    avg_dist = distances[:, 1: K_actual + 1].mean(axis=1)
    return 1.0 / (avg_dist + 1e-8)


def _typicality_adaptive_k(embeddings: np.ndarray) -> np.ndarray:
    """
    MODIFICATION: Adaptive-K Typicality.

    Uses K proportional to the cluster size instead of a fixed K=20.
        K = max(3, min(20, cluster_size // 5))

    Rationale: For small clusters, K=20 may draw neighbours from
    outside the cluster, biasing the density estimate. Scaling K
    to the cluster size keeps the neighbourhood local and accurate.
    """
    N = len(embeddings)
    if N <= 1:
        return np.ones(N)
    K = max(3, min(20, N // 5))
    K_actual = min(K, N - 1)

    nbrs = NearestNeighbors(n_neighbors=K_actual + 1,
                            metric='euclidean', n_jobs=-1).fit(embeddings)
    distances, _ = nbrs.kneighbors(embeddings)
    avg_dist = distances[:, 1: K_actual + 1].mean(axis=1)
    return 1.0 / (avg_dist + 1e-8)


# ── Shared clustering + selection logic ───────────────────────────────────────

def _cluster_and_select(embeddings:       np.ndarray,
                        budget:           int,
                        labeled_indices:  set,
                        typicality_fn,           # callable: (emb_array) → typ_array
                        max_clusters:     int = 500,
                        random_state:     int = 42) -> list:
    """
    Core of Algorithm 1.

    1. Cluster all embeddings into min(|L|+B, max_clusters) clusters.
    2. Find uncovered clusters (none of their points is already labeled).
    3. From the B largest uncovered clusters, pick the highest-typicality point.

    Parameters
    ----------
    embeddings      : (N, D) L2-normalised feature array
    budget          : B — number of points to query
    labeled_indices : set of already-labeled indices
    typicality_fn   : function(cluster_emb) → typicality scores
    max_clusters    : cap on number of clusters (paper uses 500 for CIFAR)
    random_state    : for reproducibility

    Returns
    -------
    query_indices : list of int, length ≤ budget
    """
    N = len(embeddings)
    n_labeled  = len(labeled_indices)
    n_clusters = min(n_labeled + budget, max_clusters)

    # ── Step 2: clustering for diversity ──────────────────────────────────────
    if n_clusters <= 50:
        km = KMeans(n_clusters=n_clusters,
                    random_state=random_state, n_init=10)
    else:
        km = MiniBatchKMeans(n_clusters=n_clusters,
                             random_state=random_state,
                             n_init=3, batch_size=1024)

    cluster_ids = km.fit_predict(embeddings)

    # Which clusters already contain a labeled point?
    labeled_list     = list(labeled_indices)
    labeled_clusters = set(cluster_ids[labeled_list]) if labeled_list else set()
    uncovered        = [c for c in range(n_clusters) if c not in labeled_clusters]

    # Sort uncovered clusters by size (largest first)
    cluster_sizes    = {c: int(np.sum(cluster_ids == c)) for c in uncovered}
    top_clusters     = sorted(uncovered,
                              key=lambda c: cluster_sizes[c],
                              reverse=True)[:budget]

    # ── Step 3: pick most typical point per cluster ───────────────────────────
    query_indices = []
    for c in top_clusters:
        cluster_idxs = [i for i in np.where(cluster_ids == c)[0]
                        if i not in labeled_indices]
        if not cluster_idxs:
            continue

        cluster_embs = embeddings[cluster_idxs]
        typ          = typicality_fn(cluster_embs)
        best_local   = int(np.argmax(typ))
        query_indices.append(cluster_idxs[best_local])

    # Edge case: fill remaining budget with random unlabeled points
    pool = list(set(range(N)) - labeled_indices - set(query_indices))
    random.shuffle(pool)
    while len(query_indices) < budget and pool:
        query_indices.append(pool.pop())

    return query_indices


# ── Public selector classes ───────────────────────────────────────────────────

class TPCRPSelector:
    """
    TPCRP (original): Algorithm 1 with fixed K=20 for typicality.

    Parameters
    ----------
    K            : number of neighbours for typicality (paper default 20)
    max_clusters : cluster cap (paper uses 500 for CIFAR)
    seed         : random seed
    """

    def __init__(self, K: int = 20, max_clusters: int = 500, seed: int = 42):
        self.K            = K
        self.max_clusters = max_clusters
        self.seed         = seed

    def select(self,
               embeddings:      np.ndarray,
               budget:          int,
               labeled_indices: set = None,
               **kwargs) -> list:
        """
        Parameters
        ----------
        embeddings      : (N, 512) L2-normalised feature array
        budget          : number of points to query
        labeled_indices : set of already-labeled indices

        Returns
        -------
        list of selected indices
        """
        if labeled_indices is None:
            labeled_indices = set()

        typicality_fn = lambda emb: _typicality_fixed_k(emb, K=self.K)

        return _cluster_and_select(
            embeddings      = embeddings,
            budget          = budget,
            labeled_indices = labeled_indices,
            typicality_fn   = typicality_fn,
            max_clusters    = self.max_clusters,
            random_state    = self.seed,
        )


class TPCRPModifiedSelector:
    """
    TPCRP (MODIFIED): Adaptive-K Typicality.

    Modification: Instead of a fixed K=20, K is scaled to the
    size of each cluster:
        K = max(3, min(20, cluster_size // 5))

    This avoids noisy density estimates in small clusters where
    fixed-K nearest neighbours may cross cluster boundaries.

    Parameters
    ----------
    max_clusters : cluster cap
    seed         : random seed
    """

    def __init__(self, max_clusters: int = 500, seed: int = 42):
        self.max_clusters = max_clusters
        self.seed         = seed

    def select(self,
               embeddings:      np.ndarray,
               budget:          int,
               labeled_indices: set = None,
               **kwargs) -> list:
        if labeled_indices is None:
            labeled_indices = set()

        return _cluster_and_select(
            embeddings      = embeddings,
            budget          = budget,
            labeled_indices = labeled_indices,
            typicality_fn   = _typicality_adaptive_k,
            max_clusters    = self.max_clusters,
            random_state    = self.seed,
        )
