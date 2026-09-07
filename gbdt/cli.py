"""``gbdt`` command-line interface: train, predict, evaluate, compare."""

from __future__ import annotations

import argparse
import pickle
import sys

import numpy as np

from .booster import GradientBoostedTrees
from .datasets import make_friedman1, make_moons, train_val_test_split


def _load_csv(path: str, target: str):
    import csv

    with open(path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = [list(map(float, row)) for row in reader]
    data = np.array(rows, dtype=np.float64)
    if target not in header:
        raise ValueError(f"target column {target!r} not found in header {header}")
    target_idx = header.index(target)
    feature_idx = [i for i in range(len(header)) if i != target_idx]
    X = data[:, feature_idx]
    y = data[:, target_idx]
    feature_names = [header[i] for i in feature_idx]
    return X, y, feature_names


def _get_dataset(name: str, n_samples: int, seed: int):
    if name == "friedman1":
        return make_friedman1(n_samples=n_samples, random_state=seed), "regression"
    if name == "moons":
        return make_moons(n_samples=n_samples, random_state=seed), "binary"
    raise ValueError(f"Unknown built-in dataset {name!r}")


def _add_hyperparam_args(p: argparse.ArgumentParser):
    p.add_argument("--n-estimators", type=int, default=100)
    p.add_argument("--learning-rate", type=float, default=0.1)
    p.add_argument("--max-depth", type=int, default=3)
    p.add_argument("--min-samples-leaf", type=int, default=20)
    p.add_argument("--reg-lambda", type=float, default=1.0)
    p.add_argument("--gamma", type=float, default=0.0)
    p.add_argument("--subsample", type=float, default=1.0)
    p.add_argument("--colsample", type=float, default=1.0)
    p.add_argument("--max-bins", type=int, default=32)
    p.add_argument("--early-stopping-rounds", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)


def _build_model(args, objective: str) -> GradientBoostedTrees:
    return GradientBoostedTrees(
        objective=objective,
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        reg_lambda=args.reg_lambda,
        gamma=args.gamma,
        subsample=args.subsample,
        colsample=args.colsample,
        max_bins=args.max_bins,
        early_stopping_rounds=args.early_stopping_rounds,
        random_state=args.seed,
    )


def cmd_train(args):
    if args.csv:
        X, y, _ = _load_csv(args.csv, args.target)
        objective = args.objective or "regression"
    else:
        (X, y), objective = _get_dataset(args.dataset, args.n_samples, args.seed)
        if args.objective:
            objective = args.objective

    X_train, y_train, X_val, y_val, X_test, y_test = train_val_test_split(
        X, y, random_state=args.seed
    )
    model = _build_model(args, objective)
    eval_set = (X_val, y_val) if args.early_stopping_rounds else None
    model.fit(X_train, y_train, eval_set=eval_set, verbose=args.verbose)

    test_metric = model._metric(y_test, model.predict_raw(X_test))
    metric_name = "RMSE" if objective == "regression" else "accuracy"
    n_trees = len(model.trees_)
    print(f"Trained {n_trees} trees. Test {metric_name}: {test_metric:.4f}")

    with open(args.out, "wb") as f:
        pickle.dump(model, f)
    print(f"Model saved to {args.out}")


def cmd_predict(args):
    with open(args.model, "rb") as f:
        model: GradientBoostedTrees = pickle.load(f)

    import csv

    with open(args.csv, newline="") as f:
        reader = csv.reader(f)
        next(reader)  # header row (unused: predict assumes matching column order)
        rows = [list(map(float, row)) for row in reader]
    X = np.array(rows, dtype=np.float64)
    if model.objective == "regression":
        preds = model.predict(X)
    else:
        preds = model.predict_proba(X)[:, 1]

    with open(args.out, "w") as f:
        f.write("prediction\n")
        for p in preds:
            f.write(f"{p}\n")
    print(f"Wrote {len(preds)} predictions to {args.out}")


def cmd_evaluate(args):
    (X, y), objective = _get_dataset(args.dataset, args.n_samples, args.seed)
    if args.objective:
        objective = args.objective
    X_train, y_train, X_val, y_val, X_test, y_test = train_val_test_split(
        X, y, random_state=args.seed
    )
    model = _build_model(args, objective)
    eval_set = (X_val, y_val) if args.early_stopping_rounds else None
    model.fit(X_train, y_train, eval_set=eval_set, verbose=args.verbose)

    metric_name = "RMSE" if objective == "regression" else "accuracy"
    train_metric = model._metric(y_train, model.predict_raw(X_train))
    test_metric = model._metric(y_test, model.predict_raw(X_test))
    n_trees = len(model.trees_)
    print(f"dataset={args.dataset} objective={objective} trees={n_trees}")
    print(f"train {metric_name}: {train_metric:.4f}")
    print(f"test  {metric_name}: {test_metric:.4f}")
    if model.best_iteration_ is not None and eval_set is not None:
        print(f"best_iteration (early stopping): {model.best_iteration_}")


def cmd_compare(args):
    (X, y), objective = _get_dataset(args.dataset, args.n_samples, args.seed)
    X_train, y_train, X_val, y_val, X_test, y_test = train_val_test_split(
        X, y, random_state=args.seed
    )
    metric_name = "RMSE" if objective == "regression" else "accuracy"

    gbdt = _build_model(args, objective)
    gbdt.fit(X_train, y_train)
    gbdt_metric = gbdt._metric(y_test, gbdt.predict_raw(X_test))

    # Single-tree "CART" baseline: one Newton-boosted tree with full
    # learning rate and no L2 regularization. With reg_lambda=0 and a
    # constant Hessian (squared-error/regression case), the gain formula
    # in gbdt/tree.py reduces exactly to CART's sum-of-squared-error
    # reduction criterion, so this is a faithful single-tree CART
    # baseline built from the same split-finding code, not a separate
    # implementation imported from another repo.
    baseline = GradientBoostedTrees(
        objective=objective,
        n_estimators=1,
        learning_rate=1.0,
        max_depth=args.baseline_max_depth,
        min_samples_leaf=args.min_samples_leaf,
        reg_lambda=0.0,
        gamma=0.0,
        random_state=args.seed,
    )
    baseline.fit(X_train, y_train)
    baseline_metric = baseline._metric(y_test, baseline.predict_raw(X_test))

    print(f"dataset={args.dataset} objective={objective}")
    print(
        f"single CART tree (depth={args.baseline_max_depth}):"
        f" test {metric_name} = {baseline_metric:.4f}"
    )
    print(
        f"GBDT ({args.n_estimators} trees, depth={args.max_depth}, "
        f"lr={args.learning_rate}): test {metric_name} = {gbdt_metric:.4f}"
    )
    if objective == "regression":
        improvement = (baseline_metric - gbdt_metric) / baseline_metric * 100
        print(f"GBDT reduces RMSE by {improvement:.1f}% vs the single-tree baseline")
    else:
        improvement = (gbdt_metric - baseline_metric) * 100
        print(f"GBDT improves accuracy by {improvement:.1f} points vs the single-tree baseline")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="gbdt", description="Histogram-based gradient boosted trees, from scratch.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="Train a model and save it to disk.")
    p_train.add_argument("--dataset", choices=["friedman1", "moons"], default="friedman1")
    p_train.add_argument("--csv", type=str, default=None, help="Train on a CSV file instead of a built-in dataset.")
    p_train.add_argument("--target", type=str, default=None, help="Target column name (required with --csv).")
    p_train.add_argument("--objective", choices=["regression", "binary"], default=None)
    p_train.add_argument("--n-samples", type=int, default=2000)
    p_train.add_argument("--out", type=str, default="model.pkl")
    p_train.add_argument("--verbose", action="store_true")
    _add_hyperparam_args(p_train)
    p_train.set_defaults(func=cmd_train)

    p_predict = sub.add_parser("predict", help="Predict with a saved model on a feature-only CSV.")
    p_predict.add_argument("--model", type=str, required=True)
    p_predict.add_argument("--csv", type=str, required=True)
    p_predict.add_argument("--out", type=str, default="predictions.csv")
    p_predict.set_defaults(func=cmd_predict)

    p_eval = sub.add_parser("evaluate", help="Train and report train/test metrics on a built-in dataset.")
    p_eval.add_argument("--dataset", choices=["friedman1", "moons"], default="friedman1")
    p_eval.add_argument("--objective", choices=["regression", "binary"], default=None)
    p_eval.add_argument("--n-samples", type=int, default=2000)
    p_eval.add_argument("--verbose", action="store_true")
    _add_hyperparam_args(p_eval)
    p_eval.set_defaults(func=cmd_evaluate)

    p_cmp = sub.add_parser("compare", help="Compare the GBDT ensemble against a single CART-style tree baseline.")
    p_cmp.add_argument("--dataset", choices=["friedman1", "moons"], default="friedman1")
    p_cmp.add_argument("--n-samples", type=int, default=2000)
    p_cmp.add_argument("--baseline-max-depth", type=int, default=6)
    _add_hyperparam_args(p_cmp)
    p_cmp.set_defaults(func=cmd_compare)

    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    sys.exit(main())
