"""GradientBoostedTrees: the ensemble that ties everything together.

Fits an additive model F(x) = init_score + learning_rate * sum_m tree_m(x)
by, at each stage m, fitting a new regression tree to the gradients (and
Hessians, for the Newton leaf values) of the loss evaluated at the current
ensemble output -- the standard "boosting reduces to fitting the negative
gradient" view, generalized to second order.

Supports:
  - regression (squared error) and binary classification (logistic loss)
  - shrinkage (``learning_rate``)
  - stochastic row subsampling (``subsample``) a la stochastic gradient
    boosting, decorrelating trees the way bagging does for random forests
  - random feature subsampling per split (``colsample``)
  - early stopping on a held-out validation set
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .histogram import FeatureBinner
from .losses import Loss, get_loss
from .tree import RegressionTree


@dataclass
class TrainingHistory:
    train_loss: list[float] = field(default_factory=list)
    train_metric: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    val_metric: list[float] = field(default_factory=list)


class GradientBoostedTrees:
    def __init__(
        self,
        objective: str = "regression",
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 3,
        min_samples_leaf: int = 20,
        min_child_hess: float = 1e-3,
        reg_lambda: float = 1.0,
        gamma: float = 0.0,
        subsample: float = 1.0,
        colsample: float = 1.0,
        max_bins: int = 32,
        early_stopping_rounds: int | None = None,
        random_state: int | None = None,
    ):
        if objective not in ("regression", "binary"):
            raise ValueError("objective must be 'regression' or 'binary'")
        if not (0.0 < subsample <= 1.0):
            raise ValueError("subsample must be in (0, 1]")
        if not (0.0 < colsample <= 1.0):
            raise ValueError("colsample must be in (0, 1]")

        self.objective = objective
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_child_hess = min_child_hess
        self.reg_lambda = reg_lambda
        self.gamma = gamma
        self.subsample = subsample
        self.colsample = colsample
        self.max_bins = max_bins
        self.early_stopping_rounds = early_stopping_rounds
        self.random_state = random_state

        self.trees_: list[RegressionTree] = []
        self.loss_: Loss | None = None
        self.init_score_: float = 0.0
        self.best_iteration_: int | None = None
        self.history_ = TrainingHistory()
        self.n_features_: int | None = None

    # -- metric used for early-stopping / reporting alongside loss -------
    def _metric(self, y: np.ndarray, raw_pred: np.ndarray) -> float:
        if self.objective == "regression":
            pred = raw_pred
            return float(np.sqrt(np.mean((y - pred) ** 2)))  # RMSE
        proba = self.loss_.link(raw_pred)
        pred_label = (proba >= 0.5).astype(y.dtype)
        return float(np.mean(pred_label == y))  # accuracy

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        eval_set: tuple[np.ndarray, np.ndarray] | None = None,
        verbose: bool = False,
    ) -> "GradientBoostedTrees":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError("X must be 2D")
        if len(X) != len(y):
            raise ValueError("X and y must have the same number of rows")
        if self.objective == "binary" and not set(np.unique(y)) <= {0.0, 1.0}:
            raise ValueError("binary objective requires y in {0, 1}")

        rng = np.random.default_rng(self.random_state)
        n_samples, n_features = X.shape
        self.n_features_ = n_features
        self.loss_ = get_loss(
            "squared_error" if self.objective == "regression" else "log_loss"
        )

        binner = FeatureBinner(max_bins=self.max_bins)
        binned_X = binner.fit_transform(X)

        self.init_score_ = self.loss_.init_score(y)
        raw_pred = np.full(n_samples, self.init_score_, dtype=np.float64)

        has_val = eval_set is not None
        if has_val:
            X_val, y_val = eval_set
            X_val = np.asarray(X_val, dtype=np.float64)
            y_val = np.asarray(y_val, dtype=np.float64)
            raw_val = np.full(len(X_val), self.init_score_, dtype=np.float64)

        max_features = (
            max(1, int(round(self.colsample * n_features)))
            if self.colsample < 1.0
            else None
        )
        subsample_size = (
            max(1, int(round(self.subsample * n_samples)))
            if self.subsample < 1.0
            else n_samples
        )

        self.trees_ = []
        self.history_ = TrainingHistory()
        best_val_loss = np.inf
        best_iteration = -1
        rounds_no_improve = 0

        for m in range(self.n_estimators):
            grad = self.loss_.gradient(y, raw_pred)
            hess = self.loss_.hessian(y, raw_pred)

            if self.subsample < 1.0:
                sample_indices = rng.choice(n_samples, size=subsample_size, replace=False)
            else:
                sample_indices = np.arange(n_samples)

            tree = RegressionTree(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                min_child_hess=self.min_child_hess,
                reg_lambda=self.reg_lambda,
                gamma=self.gamma,
                max_features=max_features,
                random_state=rng,
            )
            tree.fit(binned_X, binner.bin_edges_, grad, hess, sample_indices)
            self.trees_.append(tree)

            raw_pred = raw_pred + self.learning_rate * tree.predict(X)
            train_loss = self.loss_.loss(y, raw_pred)
            train_metric = self._metric(y, raw_pred)
            self.history_.train_loss.append(train_loss)
            self.history_.train_metric.append(train_metric)

            if has_val:
                raw_val = raw_val + self.learning_rate * tree.predict(X_val)
                val_loss = self.loss_.loss(y_val, raw_val)
                val_metric = self._metric(y_val, raw_val)
                self.history_.val_loss.append(val_loss)
                self.history_.val_metric.append(val_metric)

                if val_loss < best_val_loss - 1e-12:
                    best_val_loss = val_loss
                    best_iteration = m
                    rounds_no_improve = 0
                else:
                    rounds_no_improve += 1

                if verbose:
                    print(
                        f"[{m}] train_loss={train_loss:.5f} val_loss={val_loss:.5f}"
                    )

                if (
                    self.early_stopping_rounds is not None
                    and rounds_no_improve >= self.early_stopping_rounds
                ):
                    self.trees_ = self.trees_[: best_iteration + 1]
                    self.history_.train_loss = self.history_.train_loss[: best_iteration + 1]
                    self.history_.train_metric = self.history_.train_metric[: best_iteration + 1]
                    self.history_.val_loss = self.history_.val_loss[: best_iteration + 1]
                    self.history_.val_metric = self.history_.val_metric[: best_iteration + 1]
                    self.best_iteration_ = best_iteration
                    return self
            elif verbose:
                print(f"[{m}] train_loss={train_loss:.5f}")

        self.best_iteration_ = (
            best_iteration if has_val and best_iteration >= 0 else len(self.trees_) - 1
        )
        return self

    # -- prediction ----------------------------------------------------

    def predict_raw(self, X: np.ndarray) -> np.ndarray:
        if not self.trees_:
            raise RuntimeError("Model is not fit yet")
        X = np.asarray(X, dtype=np.float64)
        raw = np.full(len(X), self.init_score_, dtype=np.float64)
        for tree in self.trees_:
            raw = raw + self.learning_rate * tree.predict(X)
        return raw

    def predict(self, X: np.ndarray) -> np.ndarray:
        raw = self.predict_raw(X)
        if self.objective == "regression":
            return raw
        proba = self.loss_.link(raw)
        return (proba >= 0.5).astype(np.int64)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.objective != "binary":
            raise ValueError("predict_proba is only defined for objective='binary'")
        p1 = self.loss_.link(self.predict_raw(X))
        return np.column_stack([1 - p1, p1])

    def staged_predict_raw(self, X: np.ndarray):
        """Yield the raw prediction after each successive tree is added."""
        X = np.asarray(X, dtype=np.float64)
        raw = np.full(len(X), self.init_score_, dtype=np.float64)
        for tree in self.trees_:
            raw = raw + self.learning_rate * tree.predict(X)
            yield raw.copy()

    # -- introspection ---------------------------------------------------

    def feature_importances_gain(self) -> np.ndarray:
        """Total split gain attributed to each feature, across all trees."""
        if self.n_features_ is None:
            raise RuntimeError("Model is not fit yet")
        importances = np.zeros(self.n_features_)
        for tree in self.trees_:
            for feat_idx, gain in tree.feature_gain_.items():
                importances[feat_idx] += gain
        total = importances.sum()
        return importances / total if total > 0 else importances
