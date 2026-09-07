import numpy as np
import pytest

from gbdt.histogram import FeatureBinner, build_histogram


class TestFeatureBinner:
    def test_edges_are_sorted_and_unique(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(500, 3))
        binner = FeatureBinner(max_bins=16).fit(X)
        for edges in binner.bin_edges_:
            assert np.all(np.diff(edges) > 0)

    def test_n_bins_at_most_max_bins(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(500, 4))
        binner = FeatureBinner(max_bins=8).fit(X)
        for j in range(4):
            assert binner.n_bins(j) <= 8

    def test_constant_feature_collapses_to_a_single_populated_bin(self):
        # Every percentile of a constant column equals that constant, so
        # np.unique collapses them to exactly one edge (not zero) -- but
        # since every value ties that edge, side="right" places every row
        # in the *upper* of the resulting two bins, leaving the lower bin
        # permanently empty (and hence never chosen as a split, since an
        # empty child can never satisfy min_samples_leaf >= 1).
        X = np.column_stack([np.full(100, 7.0), np.arange(100, dtype=float)])
        binner = FeatureBinner(max_bins=10).fit(X)
        assert len(binner.bin_edges_[0]) == 1
        assert binner.n_bins(0) == 2
        binned = binner.transform(X)
        assert np.all(binned[:, 0] == 1)

    def test_transform_bin_boundary_matches_docstring_invariant(self):
        # bin(x) <= t  <=>  x < edges[t]   (see histogram.py docstring)
        rng = np.random.default_rng(4)
        X = rng.uniform(0, 100, size=(300, 1))
        binner = FeatureBinner(max_bins=10).fit(X)
        binned = binner.transform(X)
        edges = binner.bin_edges_[0]
        for t in range(len(edges)):
            left = binned[:, 0] <= t
            right_by_value = X[:, 0] < edges[t]
            np.testing.assert_array_equal(left, right_by_value.ravel())

    def test_transform_raises_on_unfit_binner(self):
        with pytest.raises(RuntimeError):
            FeatureBinner().transform(np.zeros((5, 2)))

    def test_transform_raises_on_feature_count_mismatch(self):
        binner = FeatureBinner().fit(np.zeros((10, 3)))
        with pytest.raises(ValueError):
            binner.transform(np.zeros((10, 2)))

    def test_fit_transform_matches_separate_calls(self):
        rng = np.random.default_rng(5)
        X = rng.normal(size=(200, 2))
        b1 = FeatureBinner(max_bins=12)
        combined = b1.fit_transform(X)
        b2 = FeatureBinner(max_bins=12).fit(X)
        separate = b2.transform(X)
        np.testing.assert_array_equal(combined, separate)

    def test_rejects_too_few_bins(self):
        with pytest.raises(ValueError):
            FeatureBinner(max_bins=1)

    def test_binned_values_are_within_valid_range(self):
        rng = np.random.default_rng(6)
        X = rng.normal(size=(400, 5))
        binner = FeatureBinner(max_bins=20).fit(X)
        binned = binner.transform(X)
        for j in range(5):
            assert binned[:, j].min() >= 0
            assert binned[:, j].max() < binner.n_bins(j)

    def test_out_of_sample_values_clip_into_extreme_bins(self):
        X = np.arange(100, dtype=float).reshape(-1, 1)
        binner = FeatureBinner(max_bins=10).fit(X)
        far_low = np.array([[-1000.0]])
        far_high = np.array([[1000.0]])
        assert binner.transform(far_low)[0, 0] == 0
        assert binner.transform(far_high)[0, 0] == binner.n_bins(0) - 1


class TestBuildHistogram:
    def test_sums_match_manual_computation(self):
        binned_col = np.array([0, 0, 1, 2, 2, 2])
        grad = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        hess = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        grad_hist, hess_hist, count_hist = build_histogram(binned_col, grad, hess, n_bins=3)
        np.testing.assert_allclose(grad_hist, [3.0, 3.0, 15.0])
        np.testing.assert_allclose(hess_hist, [2.0, 1.0, 3.0])
        np.testing.assert_allclose(count_hist, [2.0, 1.0, 3.0])

    def test_empty_bins_are_zero_not_missing(self):
        binned_col = np.array([0, 0, 4])
        grad = np.array([1.0, 1.0, 1.0])
        hess = np.array([1.0, 1.0, 1.0])
        grad_hist, hess_hist, count_hist = build_histogram(binned_col, grad, hess, n_bins=5)
        assert len(grad_hist) == 5
        np.testing.assert_allclose(grad_hist[1:4], [0.0, 0.0, 0.0])

    def test_total_count_matches_input_length(self):
        rng = np.random.default_rng(7)
        binned_col = rng.integers(0, 10, size=1000)
        grad = rng.normal(size=1000)
        hess = np.ones(1000)
        _, _, count_hist = build_histogram(binned_col, grad, hess, n_bins=10)
        assert count_hist.sum() == 1000
