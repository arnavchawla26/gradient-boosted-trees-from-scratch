import numpy as np
import pytest

from gbdt.booster import GradientBoostedTrees
from gbdt.datasets import make_friedman1, make_moons, train_val_test_split


class TestValidation:
    def test_rejects_bad_objective(self):
        with pytest.raises(ValueError):
            GradientBoostedTrees(objective="multiclass")

    def test_rejects_bad_subsample(self):
        with pytest.raises(ValueError):
            GradientBoostedTrees(subsample=1.5)
        with pytest.raises(ValueError):
            GradientBoostedTrees(subsample=0.0)

    def test_rejects_bad_colsample(self):
        with pytest.raises(ValueError):
            GradientBoostedTrees(colsample=0.0)

    def test_binary_objective_rejects_non_binary_labels(self):
        X = np.random.default_rng(0).normal(size=(20, 2))
        y = np.array([0, 1, 2] * 6 + [0, 1])
        model = GradientBoostedTrees(objective="binary")
        with pytest.raises(ValueError):
            model.fit(X, y)

    def test_predict_before_fit_raises(self):
        model = GradientBoostedTrees()
        with pytest.raises(RuntimeError):
            model.predict_raw(np.zeros((3, 2)))

    def test_predict_proba_requires_binary_objective(self):
        X, y = make_friedman1(n_samples=100, random_state=0)
        model = GradientBoostedTrees(objective="regression", n_estimators=3).fit(X, y)
        with pytest.raises(ValueError):
            model.predict_proba(X)


class TestRegression:
    def test_beats_naive_mean_baseline(self):
        X, y = make_friedman1(n_samples=1200, random_state=0)
        Xtr, ytr, Xv, yv, Xte, yte = train_val_test_split(X, y, random_state=0)
        model = GradientBoostedTrees(
            objective="regression", n_estimators=60, learning_rate=0.1,
            max_depth=3, random_state=0,
        ).fit(Xtr, ytr)

        naive_pred = np.full_like(yte, ytr.mean())
        naive_rmse = np.sqrt(np.mean((yte - naive_pred) ** 2))
        model_rmse = np.sqrt(np.mean((yte - model.predict(Xte)) ** 2))
        assert model_rmse < naive_rmse * 0.6

    def test_training_loss_decreases_monotonically_without_subsampling(self):
        X, y = make_friedman1(n_samples=400, random_state=1)
        model = GradientBoostedTrees(
            objective="regression", n_estimators=40, learning_rate=0.1,
            max_depth=3, subsample=1.0, random_state=1,
        ).fit(X, y)
        losses = model.history_.train_loss
        # Small learning rate + full-batch Newton boosting should not
        # increase training loss between consecutive rounds.
        diffs = np.diff(losses)
        assert np.all(diffs <= 1e-8)

    def test_more_trees_generally_improves_fit(self):
        X, y = make_friedman1(n_samples=600, random_state=2)
        Xtr, ytr, Xv, yv, Xte, yte = train_val_test_split(X, y, random_state=2)
        few = GradientBoostedTrees(objective="regression", n_estimators=5, max_depth=3, random_state=2).fit(Xtr, ytr)
        many = GradientBoostedTrees(objective="regression", n_estimators=80, max_depth=3, random_state=2).fit(Xtr, ytr)
        few_rmse = np.sqrt(np.mean((yte - few.predict(Xte)) ** 2))
        many_rmse = np.sqrt(np.mean((yte - many.predict(Xte)) ** 2))
        assert many_rmse < few_rmse


class TestClassification:
    def test_beats_majority_class_baseline(self):
        X, y = make_moons(n_samples=1000, random_state=0)
        Xtr, ytr, Xv, yv, Xte, yte = train_val_test_split(X, y, random_state=0)
        model = GradientBoostedTrees(
            objective="binary", n_estimators=50, learning_rate=0.2,
            max_depth=3, random_state=0,
        ).fit(Xtr, ytr)

        majority_acc = max(np.mean(yte == 0), np.mean(yte == 1))
        model_acc = np.mean(model.predict(Xte) == yte)
        assert model_acc > majority_acc
        assert model_acc > 0.85

    def test_predict_proba_shape_and_normalization(self):
        X, y = make_moons(n_samples=300, random_state=0)
        model = GradientBoostedTrees(objective="binary", n_estimators=20, random_state=0).fit(X, y)
        proba = model.predict_proba(X)
        assert proba.shape == (300, 2)
        np.testing.assert_allclose(proba.sum(axis=1), np.ones(300), atol=1e-9)
        assert np.all((proba >= 0) & (proba <= 1))

    def test_predict_matches_thresholded_proba(self):
        X, y = make_moons(n_samples=300, random_state=1)
        model = GradientBoostedTrees(objective="binary", n_estimators=20, random_state=1).fit(X, y)
        proba = model.predict_proba(X)[:, 1]
        preds = model.predict(X)
        np.testing.assert_array_equal(preds, (proba >= 0.5).astype(np.int64))


class TestEarlyStopping:
    def test_stops_before_n_estimators_when_overfitting(self):
        # Very small, noisy training set + high learning rate + deep trees
        # should overfit and start hurting validation loss quickly.
        X, y = make_friedman1(n_samples=150, noise=5.0, random_state=3)
        Xtr, ytr, Xv, yv, Xte, yte = train_val_test_split(X, y, random_state=3)
        model = GradientBoostedTrees(
            objective="regression", n_estimators=500, learning_rate=0.3,
            max_depth=6, min_samples_leaf=2, early_stopping_rounds=5,
            random_state=3,
        )
        model.fit(Xtr, ytr, eval_set=(Xv, yv))
        assert len(model.trees_) < 500
        assert model.best_iteration_ is not None

    def test_history_lengths_match_after_early_stopping(self):
        X, y = make_friedman1(n_samples=150, noise=5.0, random_state=3)
        Xtr, ytr, Xv, yv, Xte, yte = train_val_test_split(X, y, random_state=3)
        model = GradientBoostedTrees(
            objective="regression", n_estimators=500, learning_rate=0.3,
            max_depth=6, min_samples_leaf=2, early_stopping_rounds=5,
            random_state=3,
        )
        model.fit(Xtr, ytr, eval_set=(Xv, yv))
        n = len(model.trees_)
        assert len(model.history_.train_loss) == n
        assert len(model.history_.val_loss) == n

    def test_no_early_stopping_uses_all_estimators(self):
        X, y = make_friedman1(n_samples=200, random_state=4)
        model = GradientBoostedTrees(objective="regression", n_estimators=25, random_state=4).fit(X, y)
        assert len(model.trees_) == 25


class TestStochasticity:
    def test_subsample_reproducible_with_same_seed(self):
        X, y = make_friedman1(n_samples=300, random_state=5)
        m1 = GradientBoostedTrees(objective="regression", n_estimators=15, subsample=0.7, random_state=42).fit(X, y)
        m2 = GradientBoostedTrees(objective="regression", n_estimators=15, subsample=0.7, random_state=42).fit(X, y)
        np.testing.assert_array_equal(m1.predict(X), m2.predict(X))

    def test_subsample_differs_with_different_seed(self):
        X, y = make_friedman1(n_samples=300, random_state=5)
        m1 = GradientBoostedTrees(objective="regression", n_estimators=15, subsample=0.5, random_state=1).fit(X, y)
        m2 = GradientBoostedTrees(objective="regression", n_estimators=15, subsample=0.5, random_state=2).fit(X, y)
        assert not np.allclose(m1.predict(X), m2.predict(X))

    def test_colsample_still_produces_valid_model(self):
        X, y = make_friedman1(n_samples=300, random_state=6)
        model = GradientBoostedTrees(
            objective="regression", n_estimators=20, colsample=0.5, random_state=6
        ).fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (300,)
        assert np.all(np.isfinite(preds))

    def test_full_determinism_same_seed_full_pipeline(self):
        X, y = make_moons(n_samples=200, random_state=0)
        m1 = GradientBoostedTrees(objective="binary", n_estimators=10, subsample=0.8, colsample=0.8, random_state=7).fit(X, y)
        m2 = GradientBoostedTrees(objective="binary", n_estimators=10, subsample=0.8, colsample=0.8, random_state=7).fit(X, y)
        np.testing.assert_array_equal(m1.predict(X), m2.predict(X))


class TestIntrospection:
    def test_feature_importances_sum_to_one(self):
        X, y = make_friedman1(n_samples=500, random_state=8)
        model = GradientBoostedTrees(objective="regression", n_estimators=30, random_state=8).fit(X, y)
        importances = model.feature_importances_gain()
        assert importances.shape == (X.shape[1],)
        assert importances.sum() == pytest.approx(1.0, abs=1e-9)

    def test_pure_noise_features_get_near_zero_importance(self):
        X, y = make_friedman1(n_samples=1500, n_features=10, random_state=9)
        model = GradientBoostedTrees(
            objective="regression", n_estimators=60, max_depth=3, random_state=9
        ).fit(X, y)
        importances = model.feature_importances_gain()
        # Features 5..9 are pure noise in make_friedman1.
        assert importances[5:].sum() < importances[:5].sum() * 0.2

    def test_staged_predict_last_matches_predict_raw(self):
        X, y = make_friedman1(n_samples=200, random_state=10)
        model = GradientBoostedTrees(objective="regression", n_estimators=15, random_state=10).fit(X, y)
        staged = list(model.staged_predict_raw(X))
        assert len(staged) == 15
        np.testing.assert_allclose(staged[-1], model.predict_raw(X))

    def test_staged_predict_progressively_changes(self):
        X, y = make_friedman1(n_samples=200, random_state=11)
        model = GradientBoostedTrees(objective="regression", n_estimators=10, random_state=11).fit(X, y)
        staged = list(model.staged_predict_raw(X))
        # Each stage's raw prediction should differ from the previous one
        # (each tree contributes a nonzero update on a non-degenerate problem).
        for a, b in zip(staged[:-1], staged[1:]):
            assert not np.allclose(a, b)
