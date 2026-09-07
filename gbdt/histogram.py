"""Feature pre-binning and per-bin gradient/Hessian histograms.

This is what makes the tree learner "histogram-based" rather than the
classic CART approach of sorting each feature and scanning every unique
value as a candidate split. Each continuous feature is discretized once
(at booster-fit time) into at most ``max_bins`` buckets using empirical
percentiles; every tree in the ensemble reuses that fixed binning. Split
search then only has to consider O(bins) candidate thresholds per feature
instead of O(n), and building a histogram of gradient/Hessian sums per bin
is a single ``np.bincount`` call.

Bin semantics (see also ``gbdt/tree.py``): for feature j with sorted
distinct edges ``e_0 < e_1 < ... < e_{k-1}``, a value x falls into
``bin(x) = np.searchsorted(edges, x, side="right")``, i.e. bin(x) counts
how many edges are <= x. There are k+1 possible bins (0..k). Because
bin(x) is a non-decreasing step function of x that jumps exactly at each
edge, the set of candidate splits "bin <= t vs bin > t" for t in
[0, k-1] corresponds *exactly* to the raw-value rule "x < edges[t] goes
left, x >= edges[t] goes right" -- no separate raw-value bookkeeping is
needed at prediction time, only ``edges[t]``.
"""

from __future__ import annotations

import numpy as np


class FeatureBinner:
    """Discretizes each feature column into percentile-based bins.

    Fit once per booster.fit() call (not once per tree) so every tree in
    the ensemble shares an identical, cheap-to-reuse binning.
    """

    def __init__(self, max_bins: int = 32):
        if max_bins < 2:
            raise ValueError("max_bins must be >= 2")
        self.max_bins = max_bins
        self.bin_edges_: list[np.ndarray] | None = None

    def fit(self, X: np.ndarray) -> "FeatureBinner":
        X = np.asarray(X, dtype=np.float64)
        n_features = X.shape[1]
        self.bin_edges_ = []
        # max_bins buckets need at most max_bins - 1 interior edges.
        quantiles = np.linspace(0, 100, self.max_bins + 1)[1:-1]
        for j in range(n_features):
            col = X[:, j]
            edges = np.unique(np.percentile(col, quantiles))
            self.bin_edges_.append(edges)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.bin_edges_ is None:
            raise RuntimeError("FeatureBinner must be fit before transform")
        X = np.asarray(X, dtype=np.float64)
        n_samples, n_features = X.shape
        if n_features != len(self.bin_edges_):
            raise ValueError(
                f"Expected {len(self.bin_edges_)} features, got {n_features}"
            )
        binned = np.empty((n_samples, n_features), dtype=np.int32)
        for j in range(n_features):
            binned[:, j] = np.searchsorted(self.bin_edges_[j], X[:, j], side="right")
        return binned

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def n_bins(self, feature_idx: int) -> int:
        """Number of bins (0..n_bins-1) for a feature, i.e. len(edges) + 1."""
        return len(self.bin_edges_[feature_idx]) + 1


def build_histogram(
    binned_col: np.ndarray,
    grad: np.ndarray,
    hess: np.ndarray,
    n_bins: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-bin sums of gradient, Hessian, and count for one feature column.

    ``binned_col`` holds bin indices in [0, n_bins) for the rows currently
    at this tree node (already row-subsampled/row-filtered by the caller).
    """
    grad_hist = np.bincount(binned_col, weights=grad, minlength=n_bins)
    hess_hist = np.bincount(binned_col, weights=hess, minlength=n_bins)
    count_hist = np.bincount(binned_col, minlength=n_bins).astype(np.float64)
    return grad_hist, hess_hist, count_hist
