"""Deterministic synthetic datasets used by the CLI, tests, and benchmarks.

No large data files are committed to the repo -- everything here is
generated on demand from a fixed random seed, matching the pattern used
by prior projects in this portfolio (see e.g. ``kmeans-clustering-from-
scratch``'s and ``probabilistic-data-structures-toolkit``'s generators).
"""

from __future__ import annotations

import numpy as np


def make_friedman1(n_samples: int = 2000, n_features: int = 10, noise: float = 1.0, random_state: int = 0):
    """A Friedman-#1-style nonlinear regression benchmark.

    y = 10*sin(pi*x0*x1) + 20*(x2-0.5)^2 + 10*x3 + 5*x4 + noise,
    with x5..x_{n_features-1} pure noise features (irrelevant to y) so a
    good learner should down-weight them in feature_importances_gain().
    Features are uniform on [0, 1]; this is the classic benchmark used to
    demonstrate that boosted trees can capture nonlinear interactions a
    single linear model cannot.
    """
    if n_features < 5:
        raise ValueError("make_friedman1 requires n_features >= 5")
    rng = np.random.default_rng(random_state)
    X = rng.uniform(0, 1, size=(n_samples, n_features))
    y = (
        10 * np.sin(np.pi * X[:, 0] * X[:, 1])
        + 20 * (X[:, 2] - 0.5) ** 2
        + 10 * X[:, 3]
        + 5 * X[:, 4]
        + rng.normal(0, noise, size=n_samples)
    )
    return X, y


def make_moons(n_samples: int = 2000, noise: float = 0.25, random_state: int = 0):
    """Two interleaving half-moons -- a nonlinearly separable binary problem.

    A linear classifier cannot separate these classes well; a boosted-tree
    ensemble with enough depth/rounds can carve out the curved boundary.
    """
    rng = np.random.default_rng(random_state)
    n_per_class = n_samples // 2
    theta1 = rng.uniform(0, np.pi, size=n_per_class)
    x1 = np.column_stack([np.cos(theta1), np.sin(theta1)])
    theta2 = rng.uniform(0, np.pi, size=n_samples - n_per_class)
    x2 = np.column_stack([1 - np.cos(theta2), 1 - np.sin(theta2) - 0.5])

    X = np.vstack([x1, x2])
    y = np.concatenate([np.zeros(n_per_class), np.ones(n_samples - n_per_class)])
    X = X + rng.normal(0, noise, size=X.shape)

    # Shuffle so classes aren't in contiguous blocks.
    perm = rng.permutation(n_samples)
    return X[perm], y[perm]


def train_val_test_split(X, y, val_frac=0.15, test_frac=0.15, random_state=0):
    rng = np.random.default_rng(random_state)
    n = len(X)
    perm = rng.permutation(n)
    n_test = int(round(n * test_frac))
    n_val = int(round(n * val_frac))
    test_idx = perm[:n_test]
    val_idx = perm[n_test : n_test + n_val]
    train_idx = perm[n_test + n_val :]
    return (
        X[train_idx],
        y[train_idx],
        X[val_idx],
        y[val_idx],
        X[test_idx],
        y[test_idx],
    )
