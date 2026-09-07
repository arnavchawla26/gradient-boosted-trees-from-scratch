"""Loss functions for gradient boosting.

Every loss exposes the three pieces Newton-boosting needs:

- ``init_score``: the constant raw-score prediction that minimizes the loss
  before any trees are added (this becomes the ensemble's base score).
- ``gradient(y, raw_pred)``: first derivative of the loss w.r.t. the raw
  (pre-link) prediction, evaluated at the current ensemble output.
- ``hessian(y, raw_pred)``: second derivative of the loss w.r.t. the raw
  prediction. Histogram-based GBDT uses both to fit each tree's leaves via
  a Newton step: leaf_value = -sum(grad) / (sum(hess) + reg_lambda).

All losses work on the *raw* (pre-link) score. For least-squares regression
the raw score *is* the prediction. For logistic loss the raw score is the
log-odds, and ``link`` (sigmoid) turns it into a probability.
"""

from __future__ import annotations

import numpy as np


class Loss:
    """Base class documenting the interface every loss implements."""

    name = "base"

    def init_score(self, y: np.ndarray) -> float:
        raise NotImplementedError

    def gradient(self, y: np.ndarray, raw_pred: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def hessian(self, y: np.ndarray, raw_pred: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def loss(self, y: np.ndarray, raw_pred: np.ndarray) -> float:
        raise NotImplementedError

    def link(self, raw_pred: np.ndarray) -> np.ndarray:
        """Map a raw score to the loss's natural output space (identity by default)."""
        return raw_pred


class LeastSquaresLoss(Loss):
    """0.5 * (y - pred)^2 -- used for regression.

    Gradient w.r.t. pred is (pred - y); Hessian is the constant 1, which
    makes the Newton leaf value reduce to the ordinary mean of the
    negative gradients in that leaf (i.e. plain gradient-descent boosting
    on residuals), matching the textbook GBDT-for-regression derivation.
    """

    name = "squared_error"

    def init_score(self, y: np.ndarray) -> float:
        return float(np.mean(y))

    def gradient(self, y: np.ndarray, raw_pred: np.ndarray) -> np.ndarray:
        return raw_pred - y

    def hessian(self, y: np.ndarray, raw_pred: np.ndarray) -> np.ndarray:
        return np.ones_like(raw_pred, dtype=np.float64)

    def loss(self, y: np.ndarray, raw_pred: np.ndarray) -> float:
        return float(np.mean(0.5 * (y - raw_pred) ** 2))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    # Numerically stable logistic sigmoid.
    out = np.empty_like(x, dtype=np.float64)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    exp_x = np.exp(x[~pos])
    out[~pos] = exp_x / (1.0 + exp_x)
    return out


class LogisticLoss(Loss):
    """Binary cross-entropy on the log-odds (sigmoid link) -- for classification.

    y in {0, 1}. Raw score is the log-odds; p = sigmoid(raw). Standard
    Newton-boosting derivatives: grad = p - y, hess = p * (1 - p).
    """

    name = "log_loss"

    def init_score(self, y: np.ndarray) -> float:
        p = np.clip(np.mean(y), 1e-15, 1 - 1e-15)
        return float(np.log(p / (1 - p)))

    def gradient(self, y: np.ndarray, raw_pred: np.ndarray) -> np.ndarray:
        p = _sigmoid(raw_pred)
        return p - y

    def hessian(self, y: np.ndarray, raw_pred: np.ndarray) -> np.ndarray:
        p = _sigmoid(raw_pred)
        # Floor away from exactly 0 so a saturated leaf never divides by ~0.
        return np.maximum(p * (1 - p), 1e-16)

    def loss(self, y: np.ndarray, raw_pred: np.ndarray) -> float:
        p = np.clip(_sigmoid(raw_pred), 1e-15, 1 - 1e-15)
        return float(np.mean(-(y * np.log(p) + (1 - y) * np.log(1 - p))))

    def link(self, raw_pred: np.ndarray) -> np.ndarray:
        return _sigmoid(raw_pred)


LOSSES = {
    "squared_error": LeastSquaresLoss,
    "log_loss": LogisticLoss,
}


def get_loss(name: str) -> Loss:
    try:
        return LOSSES[name]()
    except KeyError as exc:
        raise ValueError(
            f"Unknown loss {name!r}; choose from {sorted(LOSSES)}"
        ) from exc
