import numpy as np

from gbdt.datasets import make_friedman1, make_moons, train_val_test_split


class TestMakeFriedman1:
    def test_shapes(self):
        X, y = make_friedman1(n_samples=250, n_features=7, random_state=0)
        assert X.shape == (250, 7)
        assert y.shape == (250,)

    def test_deterministic_with_same_seed(self):
        X1, y1 = make_friedman1(n_samples=100, random_state=42)
        X2, y2 = make_friedman1(n_samples=100, random_state=42)
        np.testing.assert_array_equal(X1, X2)
        np.testing.assert_array_equal(y1, y2)

    def test_different_seeds_differ(self):
        X1, y1 = make_friedman1(n_samples=100, random_state=1)
        X2, y2 = make_friedman1(n_samples=100, random_state=2)
        assert not np.allclose(y1, y2)

    def test_features_within_unit_interval(self):
        X, _ = make_friedman1(n_samples=500, random_state=0)
        assert X.min() >= 0.0
        assert X.max() <= 1.0

    def test_requires_at_least_five_features(self):
        import pytest

        with pytest.raises(ValueError):
            make_friedman1(n_features=3)

    def test_zero_noise_is_a_deterministic_function_of_x(self):
        X, y = make_friedman1(n_samples=50, noise=0.0, random_state=0)
        expected = (
            10 * np.sin(np.pi * X[:, 0] * X[:, 1])
            + 20 * (X[:, 2] - 0.5) ** 2
            + 10 * X[:, 3]
            + 5 * X[:, 4]
        )
        np.testing.assert_allclose(y, expected)


class TestMakeMoons:
    def test_shapes_and_labels(self):
        X, y = make_moons(n_samples=400, random_state=0)
        assert X.shape == (400, 2)
        assert set(np.unique(y)) <= {0.0, 1.0}

    def test_roughly_balanced_classes(self):
        X, y = make_moons(n_samples=1000, random_state=0)
        frac_pos = y.mean()
        assert 0.4 < frac_pos < 0.6

    def test_deterministic_with_same_seed(self):
        X1, y1 = make_moons(n_samples=200, random_state=5)
        X2, y2 = make_moons(n_samples=200, random_state=5)
        np.testing.assert_array_equal(X1, X2)
        np.testing.assert_array_equal(y1, y2)

    def test_low_noise_is_more_separable_than_high_noise(self):
        # Sanity check on the generator itself: a dependency-free
        # nearest-centroid accuracy proxy should be more favorable at low
        # noise than at high noise.
        def nc_accuracy(noise):
            X, y = make_moons(n_samples=400, noise=noise, random_state=0)
            c0 = X[y == 0].mean(axis=0)
            c1 = X[y == 1].mean(axis=0)
            d0 = np.linalg.norm(X - c0, axis=1)
            d1 = np.linalg.norm(X - c1, axis=1)
            pred = (d1 < d0).astype(float)
            return np.mean(pred == y)

        low_noise_acc = nc_accuracy(0.05)
        high_noise_acc = nc_accuracy(0.6)
        assert low_noise_acc >= high_noise_acc


class TestTrainValTestSplit:
    def test_sizes_sum_to_total(self):
        X = np.arange(1000).reshape(-1, 1).astype(float)
        y = np.arange(1000).astype(float)
        Xtr, ytr, Xv, yv, Xte, yte = train_val_test_split(X, y, val_frac=0.15, test_frac=0.15, random_state=0)
        assert len(Xtr) + len(Xv) + len(Xte) == 1000
        assert len(Xtr) == len(ytr)
        assert len(Xv) == len(yv)
        assert len(Xte) == len(yte)

    def test_approximate_fractions(self):
        X = np.arange(2000).reshape(-1, 1).astype(float)
        y = np.arange(2000).astype(float)
        Xtr, ytr, Xv, yv, Xte, yte = train_val_test_split(X, y, val_frac=0.2, test_frac=0.1, random_state=0)
        assert abs(len(Xv) - 400) <= 2
        assert abs(len(Xte) - 200) <= 2

    def test_no_row_appears_in_more_than_one_split(self):
        X = np.arange(500).reshape(-1, 1).astype(float)
        y = np.arange(500).astype(float)
        Xtr, ytr, Xv, yv, Xte, yte = train_val_test_split(X, y, random_state=0)
        train_ids = set(Xtr[:, 0].tolist())
        val_ids = set(Xv[:, 0].tolist())
        test_ids = set(Xte[:, 0].tolist())
        assert train_ids.isdisjoint(val_ids)
        assert train_ids.isdisjoint(test_ids)
        assert val_ids.isdisjoint(test_ids)
        assert train_ids | val_ids | test_ids == set(range(500))

    def test_deterministic_with_same_seed(self):
        X = np.arange(300).reshape(-1, 1).astype(float)
        y = np.arange(300).astype(float)
        split1 = train_val_test_split(X, y, random_state=3)
        split2 = train_val_test_split(X, y, random_state=3)
        for a, b in zip(split1, split2):
            np.testing.assert_array_equal(a, b)
