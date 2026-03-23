"""
selection_strategies/random_selector.py
----------------------------------------
Random selection baseline for active learning.
Uniformly samples B indices from the unlabeled pool.
"""

import random
import numpy as np


class RandomSelector:
    """
    Baseline AL strategy: uniform random sampling from unlabeled pool.

    Parameters
    ----------
    seed : int
        Random seed for reproducibility.
    """

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.rng  = random.Random(seed)

    def select(self,
               budget:          int,
               labeled_indices: set,
               n_total:         int,
               **kwargs) -> list:
        """
        Randomly sample `budget` indices from the unlabeled pool.

        Parameters
        ----------
        budget          : number of points to query
        labeled_indices : set of already-labeled indices (excluded)
        n_total         : total size of the unlabeled pool

        Returns
        -------
        List of `budget` selected indices.
        """
        pool = list(set(range(n_total)) - labeled_indices)
        return self.rng.sample(pool, min(budget, len(pool)))
