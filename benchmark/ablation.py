"""Empirical ablations backing two design claims made in the README.

1. Histogram bin count vs. accuracy/speed: the whole point of histogram-
   based split finding is trading a little split precision for a lot of
   speed by capping candidate splits per feature at ``max_bins`` instead
   of scanning every unique value. This measures how much accuracy is
   actually given up (if any) as max_bins shrinks from 255 down to 4, and
   how much wall-clock time is saved.

2. Ensemble size vs. the single-tree baseline: how many boosting rounds
   it actually takes to beat a single unregularized CART-equivalent tree,
   and whether more rounds keep helping or plateau/overfit past a point.

Run: python benchmark/ablation.py
"""

from __future__ import annotations

import time

import numpy as np

from gbdt.booster import GradientBoostedTrees
from gbdt.datasets import make_friedman1, train_val_test_split


def bin_count_ablation():
    print("=== Ablation 1: histogram bin count vs. accuracy/speed ===")
    X, y = make_friedman1(n_samples=4000, n_features=10, random_state=0)
    Xtr, ytr, Xv, yv, Xte, yte = train_val_test_split(X, y, random_state=0)

    print(f"{'max_bins':>10} {'test RMSE':>12} {'fit time (s)':>14}")
    for max_bins in (255, 64, 32, 16, 8, 4):
        start = time.perf_counter()
        model = GradientBoostedTrees(
            objective="regression", n_estimators=80, learning_rate=0.1,
            max_depth=3, max_bins=max_bins, random_state=0,
        ).fit(Xtr, ytr)
        elapsed = time.perf_counter() - start
        rmse = np.sqrt(np.mean((yte - model.predict(Xte)) ** 2))
        print(f"{max_bins:>10} {rmse:>12.4f} {elapsed:>14.3f}")
    print(
        "Interpretation (measured, not assumed): accuracy stays essentially flat "
        "from 255 bins down to 16, then degrades once bins get very coarse (4 "
        "bins loses real accuracy). Fit time, however, stays roughly constant "
        "across this whole range in this pure-Python implementation -- the cost "
        "here is dominated by per-sample gradient/Hessian evaluation and "
        "Python-level tree recursion, not the O(1)-per-feature bincount lookup "
        "that bin count actually controls. In a vectorized/compiled histogram "
        "builder (as in real GBDT libraries, where recursion overhead is "
        "negligible) capping bins would show its usual speed payoff much more "
        "clearly; here it mainly demonstrates that accuracy tolerates far fewer "
        "bins than the max_bins=32 default before it costs anything.\n"
    )


def ensemble_size_ablation():
    print("=== Ablation 2: ensemble size vs. single-tree baseline ===")
    X, y = make_friedman1(n_samples=3000, n_features=10, random_state=1)
    Xtr, ytr, Xv, yv, Xte, yte = train_val_test_split(X, y, random_state=1)

    baseline = GradientBoostedTrees(
        objective="regression", n_estimators=1, learning_rate=1.0,
        max_depth=6, reg_lambda=0.0, random_state=1,
    ).fit(Xtr, ytr)
    baseline_rmse = np.sqrt(np.mean((yte - baseline.predict(Xte)) ** 2))
    print(f"single CART-equivalent tree (depth=6): test RMSE = {baseline_rmse:.4f}\n")

    print(f"{'n_estimators':>13} {'test RMSE':>12} {'beats baseline?':>16}")
    for n_estimators in (1, 2, 5, 10, 20, 50, 100, 200, 400):
        model = GradientBoostedTrees(
            objective="regression", n_estimators=n_estimators, learning_rate=0.1,
            max_depth=3, random_state=1,
        ).fit(Xtr, ytr)
        rmse = np.sqrt(np.mean((yte - model.predict(Xte)) ** 2))
        beats = "yes" if rmse < baseline_rmse else "no"
        print(f"{n_estimators:>13} {rmse:>12.4f} {beats:>16}")


if __name__ == "__main__":
    bin_count_ablation()
    ensemble_size_ablation()
