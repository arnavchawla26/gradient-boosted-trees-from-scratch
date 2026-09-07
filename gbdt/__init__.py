"""gradient-boosted-trees-from-scratch

A from-scratch, dependency-on-ML-frameworks-free implementation of
histogram-based gradient boosted decision trees (the core algorithm behind
XGBoost / LightGBM), supporting regression and binary classification.
"""

from .booster import GradientBoostedTrees
from .losses import LeastSquaresLoss, LogisticLoss

__all__ = ["GradientBoostedTrees", "LeastSquaresLoss", "LogisticLoss"]

__version__ = "0.1.0"
