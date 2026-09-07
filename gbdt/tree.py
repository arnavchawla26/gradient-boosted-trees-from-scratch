"""A single histogram-based regression tree, Newton-boosted.

Every tree in a GBDT ensemble is a regression tree fit to the current
gradients/Hessians of the loss (not to the raw targets), using the
standard XGBoost-style exact-histogram split criterion:

    gain(split) = 0.5 * [ G_L^2/(H_L + lambda) + G_R^2/(H_R + lambda)
                           - G^2/(H + lambda) ] - gamma

and each leaf's value is the Newton step that minimizes the loss's
second-order Taylor expansion within that leaf:

    leaf_value = -G / (H + lambda)

where G, H are the sum of gradients/Hessians of the samples in that leaf
and lambda (``reg_lambda``) is an L2 regularization term on leaf weights.
``gamma`` is a minimum-gain threshold (a split is only taken if it beats
the cost of making a new leaf), which naturally handles "no split is
better than any split" the way CART's impurity-decrease check does.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from .histogram import build_histogram


@dataclass
class _Node:
    is_leaf: bool = True
    value: float = 0.0
    feature_idx: int = -1
    threshold: float = 0.0  # raw-value split: x < threshold -> left
    left: "_Node | None" = None
    right: "_Node | None" = None


class RegressionTree:
    """One Newton-boosted regression tree over pre-binned features.

    Parameters mirror the booster's tree-growth knobs so a tree can be
    grown standalone (e.g. for the CART baseline used in benchmarking).
    """

    def __init__(
        self,
        max_depth: int = 6,
        min_samples_leaf: int = 20,
        min_child_hess: float = 1e-3,
        reg_lambda: float = 1.0,
        gamma: float = 0.0,
        max_features: int | None = None,
        random_state: np.random.Generator | None = None,
    ):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_child_hess = min_child_hess
        self.reg_lambda = reg_lambda
        self.gamma = gamma
        self.max_features = max_features
        self.random_state = random_state or np.random.default_rng()
        self.root_: _Node | None = None
        self.n_leaves_: int = 0
        self.n_splits_: int = 0
        self.feature_gain_: dict[int, float] = defaultdict(float)

    # -- fitting ---------------------------------------------------------

    def fit(
        self,
        binned_X: np.ndarray,
        bin_edges: list[np.ndarray],
        grad: np.ndarray,
        hess: np.ndarray,
        sample_indices: np.ndarray | None = None,
    ) -> "RegressionTree":
        n_samples, n_features = binned_X.shape
        if sample_indices is None:
            sample_indices = np.arange(n_samples)
        self._binned_X = binned_X
        self._bin_edges = bin_edges
        self._grad = grad
        self._hess = hess
        self.n_leaves_ = 0
        self.n_splits_ = 0
        self.feature_gain_ = defaultdict(float)
        self.root_ = self._build(sample_indices, depth=0)
        return self

    def _leaf_value(self, idx: np.ndarray) -> float:
        g = self._grad[idx].sum()
        h = self._hess[idx].sum()
        return float(-g / (h + self.reg_lambda))

    def _build(self, idx: np.ndarray, depth: int) -> _Node:
        node = _Node(is_leaf=True, value=self._leaf_value(idx))
        self.n_leaves_ += 1

        if depth >= self.max_depth or len(idx) < 2 * self.min_samples_leaf:
            return node

        best = self._find_best_split(idx)
        if best is None:
            return node

        feature_idx, threshold, gain, left_idx, right_idx = best
        if gain <= self.gamma:
            return node

        self.n_leaves_ -= 1  # this node becomes internal, not a leaf
        self.n_splits_ += 1
        self.feature_gain_[feature_idx] += gain
        node.is_leaf = False
        node.feature_idx = feature_idx
        node.threshold = threshold
        node.left = self._build(left_idx, depth + 1)
        node.right = self._build(right_idx, depth + 1)
        return node

    def _find_best_split(self, idx: np.ndarray):
        n_features = self._binned_X.shape[1]
        if self.max_features is not None and self.max_features < n_features:
            candidate_features = self.random_state.choice(
                n_features, size=self.max_features, replace=False
            )
        else:
            candidate_features = range(n_features)

        grad = self._grad[idx]
        hess = self._hess[idx]
        total_grad = grad.sum()
        total_hess = hess.sum()
        parent_score = (total_grad ** 2) / (total_hess + self.reg_lambda)

        best_gain = 0.0
        best = None

        for feat in candidate_features:
            edges = self._bin_edges[feat]
            if len(edges) == 0:
                continue  # constant feature, no candidate splits
            n_bins = len(edges) + 1
            col = self._binned_X[idx, feat]
            grad_hist, hess_hist, count_hist = build_histogram(col, grad, hess, n_bins)

            cum_grad = np.cumsum(grad_hist)
            cum_hess = np.cumsum(hess_hist)
            cum_count = np.cumsum(count_hist)

            # Candidate splits t = 0..len(edges)-1: left = bins [0, t], right = rest.
            n_splits = len(edges)
            left_grad = cum_grad[:n_splits]
            left_hess = cum_hess[:n_splits]
            left_count = cum_count[:n_splits]
            right_grad = total_grad - left_grad
            right_hess = total_hess - left_hess
            right_count = len(idx) - left_count

            valid = (
                (left_count >= self.min_samples_leaf)
                & (right_count >= self.min_samples_leaf)
                & (left_hess >= self.min_child_hess)
                & (right_hess >= self.min_child_hess)
            )
            if not np.any(valid):
                continue

            # Invalid (filtered-out) splits can have a zero denominator
            # (e.g. reg_lambda=0 and an empty-Hessian bin); they're masked
            # out below regardless, so silence the resulting 0/0 warnings
            # rather than letting them leak to the caller.
            with np.errstate(divide="ignore", invalid="ignore"):
                gain = 0.5 * (
                    (left_grad ** 2) / (left_hess + self.reg_lambda)
                    + (right_grad ** 2) / (right_hess + self.reg_lambda)
                    - parent_score
                )
            gain = np.where(valid, gain, -np.inf)

            t_best = int(np.argmax(gain))
            if gain[t_best] > best_gain:
                best_gain = float(gain[t_best])
                threshold = float(edges[t_best])
                # bin <= t_best is exactly raw_value < threshold (see module
                # docstring in histogram.py for why these coincide).
                mask = col <= t_best
                left_idx = idx[mask]
                right_idx = idx[~mask]
                best = (feat, threshold, best_gain, left_idx, right_idx)

        return best

    # -- prediction --------------------------------------------------------

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        out = np.empty(X.shape[0], dtype=np.float64)
        for i in range(X.shape[0]):
            out[i] = self._predict_row(X[i])
        return out

    def _predict_row(self, row: np.ndarray) -> float:
        node = self.root_
        while not node.is_leaf:
            node = node.left if row[node.feature_idx] < node.threshold else node.right
        return node.value
