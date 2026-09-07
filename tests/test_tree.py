import numpy as np
import pytest

from gbdt.histogram import FeatureBinner
from gbdt.tree import RegressionTree


def _fit_tree(X, grad, hess, **kwargs):
    binner = FeatureBinner(max_bins=32).fit(X)
    binned = binner.transform(X)
    tree = RegressionTree(**kwargs)
    tree.fit(binned, binner.bin_edges_, grad, hess)
    return tree


def _leaf_sample_counts(node, X):
    """Recompute leaf sizes by walking the tree with its own raw-value rule."""
    counts = []

    def recurse(n, idx):
        if n.is_leaf:
            counts.append(len(idx))
            return
        left_mask = X[idx, n.feature_idx] < n.threshold
        recurse(n.left, idx[left_mask])
        recurse(n.right, idx[~left_mask])

    recurse(node, np.arange(len(X)))
    return counts


def _tree_depth(node):
    if node.is_leaf:
        return 0
    return 1 + max(_tree_depth(node.left), _tree_depth(node.right))


class TestSingleSplit:
    def test_finds_the_obvious_split(self):
        # x < 0.5 -> target -10, x >= 0.5 -> target +10. Squared-error
        # gradients/Hessians (grad = pred - y at pred=0, hess = 1) should
        # let a depth-1 tree recover this split exactly.
        rng = np.random.default_rng(0)
        n = 200
        x = rng.uniform(0, 1, size=n)
        y = np.where(x < 0.5, -10.0, 10.0)
        X = x.reshape(-1, 1)
        grad = -y  # pred starts at 0: grad = pred - y = -y
        hess = np.ones(n)

        tree = _fit_tree(X, grad, hess, max_depth=1, min_samples_leaf=5, reg_lambda=0.0)
        assert not tree.root_.is_leaf
        assert tree.root_.feature_idx == 0
        assert 0.3 < tree.root_.threshold < 0.7

        preds = tree.predict(X)
        # Newton step at a pure leaf recovers the leaf's mean target exactly
        # (grad=pred-y, hess=1, reg_lambda=0 => leaf_value = mean(y)).
        left_mask = x < tree.root_.threshold
        np.testing.assert_allclose(preds[left_mask], np.full(left_mask.sum(), -10.0), atol=1e-8)
        np.testing.assert_allclose(preds[~left_mask], np.full((~left_mask).sum(), 10.0), atol=1e-8)

    def test_leaf_value_matches_newton_formula(self):
        X = np.zeros((10, 1))  # constant feature -> no split possible, one leaf
        grad = np.array([1.0, -2.0, 3.0, -4.0, 5.0, -1.0, 2.0, -3.0, 4.0, -5.0])
        hess = np.ones(10)
        reg_lambda = 2.0
        tree = _fit_tree(X, grad, hess, max_depth=3, reg_lambda=reg_lambda)
        expected = -grad.sum() / (hess.sum() + reg_lambda)
        assert tree.root_.is_leaf
        assert tree.root_.value == pytest.approx(expected)


class TestStoppingCriteria:
    def test_respects_max_depth(self):
        rng = np.random.default_rng(1)
        X = rng.uniform(size=(400, 3))
        grad = rng.normal(size=400)
        hess = np.ones(400)
        for depth in (1, 2, 4):
            tree = _fit_tree(X, grad, hess, max_depth=depth, min_samples_leaf=1, reg_lambda=0.1)
            assert _tree_depth(tree.root_) <= depth

    def test_respects_min_samples_leaf(self):
        rng = np.random.default_rng(2)
        X = rng.uniform(size=(300, 2))
        grad = rng.normal(size=300)
        hess = np.ones(300)
        min_leaf = 40
        tree = _fit_tree(X, grad, hess, max_depth=6, min_samples_leaf=min_leaf, reg_lambda=0.1)
        leaf_sizes = _leaf_sample_counts(tree.root_, X)
        assert all(size >= min_leaf for size in leaf_sizes)

    def test_high_gamma_prevents_any_split(self):
        rng = np.random.default_rng(3)
        X = rng.uniform(size=(200, 2))
        grad = rng.normal(scale=0.01, size=200)  # tiny signal -> tiny gains
        hess = np.ones(200)
        tree = _fit_tree(X, grad, hess, max_depth=5, gamma=1e6, reg_lambda=0.1)
        assert tree.root_.is_leaf
        assert tree.n_splits_ == 0
        assert tree.n_leaves_ == 1

    def test_zero_gamma_allows_splits_when_gain_positive(self):
        rng = np.random.default_rng(4)
        n = 400
        x = rng.uniform(size=n)
        y = np.where(x < 0.5, -5.0, 5.0) + rng.normal(scale=0.01, size=n)
        X = x.reshape(-1, 1)
        grad = -y
        hess = np.ones(n)
        tree = _fit_tree(X, grad, hess, max_depth=1, gamma=0.0, reg_lambda=0.0)
        assert tree.n_splits_ == 1


class TestFeatureImportance:
    def test_only_informative_feature_gets_gain(self):
        # At max_depth=1 there is exactly one split to make, so it must go
        # to whichever feature offers the higher gain -- the informative
        # one. (At greater depth, coarse histogram binning can leave a
        # little residual signal at the leaves after the first split, which
        # a second split may spuriously "explain" using the noise feature;
        # that's a real, expected property of finite-bin histograms, not
        # tested here -- see test_histogram.py's boundary-invariant test.)
        rng = np.random.default_rng(5)
        n = 500
        x_informative = rng.uniform(size=n)
        x_noise = rng.uniform(size=n)
        y = np.where(x_informative < 0.5, -10.0, 10.0)
        X = np.column_stack([x_informative, x_noise])
        grad = -y
        hess = np.ones(n)
        tree = _fit_tree(X, grad, hess, max_depth=1, min_samples_leaf=5, reg_lambda=0.0)
        assert 0 in tree.feature_gain_
        assert tree.feature_gain_.get(1, 0.0) == 0.0

    def test_max_features_restricts_candidate_set_without_crashing(self):
        rng = np.random.default_rng(6)
        X = rng.uniform(size=(300, 10))
        grad = rng.normal(size=300)
        hess = np.ones(300)
        tree = _fit_tree(
            X, grad, hess, max_depth=4, min_samples_leaf=5, reg_lambda=0.1,
            max_features=2, random_state=np.random.default_rng(0),
        )
        assert tree.root_ is not None
        preds = tree.predict(X)
        assert preds.shape == (300,)


def test_predict_shape_and_determinism():
    rng = np.random.default_rng(7)
    X = rng.normal(size=(150, 4))
    grad = rng.normal(size=150)
    hess = np.ones(150)
    tree = _fit_tree(X, grad, hess, max_depth=3, reg_lambda=1.0)
    p1 = tree.predict(X)
    p2 = tree.predict(X)
    assert p1.shape == (150,)
    np.testing.assert_array_equal(p1, p2)


def test_sample_indices_subset_only_uses_those_rows():
    # Fit once via sample_indices on full-size arrays (with unrelated rows
    # "corrupted" outside sample_indices), and once on an already-subsetted
    # copy of the same rows using the *same* binning. If sample_indices
    # correctly restricts the fit, the two trees must make identical
    # predictions on the included rows regardless of what's in the
    # corrupted/excluded rows.
    rng = np.random.default_rng(8)
    n = 300
    X = rng.uniform(size=(n, 1))
    y_full = np.where(X[:, 0] < 0.5, -10.0, 10.0)
    y_corrupted = y_full.copy()
    excluded = np.arange(n // 2)
    included = np.arange(n // 2, n)
    y_corrupted[excluded] = 999.0

    binner = FeatureBinner(max_bins=32).fit(X)
    binned = binner.transform(X)
    grad = -y_corrupted
    hess = np.ones(n)

    tree_via_indices = RegressionTree(max_depth=2, min_samples_leaf=1, reg_lambda=0.3)
    tree_via_indices.fit(binned, binner.bin_edges_, grad, hess, sample_indices=included)

    tree_via_subset = RegressionTree(max_depth=2, min_samples_leaf=1, reg_lambda=0.3)
    tree_via_subset.fit(
        binned[included], binner.bin_edges_, grad[included], hess[included]
    )

    np.testing.assert_allclose(
        tree_via_indices.predict(X[included]), tree_via_subset.predict(X[included])
    )
    # And corruption of the excluded rows must not have leaked in: the
    # induced split still only reflects the true -10/10 pattern among the
    # included rows (verified indirectly via the subset-equivalence above,
    # plus a sanity check that predictions are close to the true pattern).
    preds = tree_via_indices.predict(X[included])
    assert np.corrcoef(preds, y_full[included])[0, 1] > 0.95


def test_two_way_interaction_requires_depth_two():
    # y depends on an XOR-like interaction of two binary-ish features;
    # a depth-1 tree cannot reduce loss much, a depth-2 tree can isolate
    # all four quadrants.
    rng = np.random.default_rng(9)
    n = 800
    x0 = rng.uniform(size=n)
    x1 = rng.uniform(size=n)
    y = np.where((x0 < 0.5) == (x1 < 0.5), -10.0, 10.0)
    X = np.column_stack([x0, x1])
    grad = -y
    hess = np.ones(n)

    shallow = _fit_tree(X, grad, hess, max_depth=1, min_samples_leaf=5, reg_lambda=0.0)
    deep = _fit_tree(X, grad, hess, max_depth=2, min_samples_leaf=5, reg_lambda=0.0)

    shallow_mse = np.mean((shallow.predict(X) - y) ** 2)
    deep_mse = np.mean((deep.predict(X) - y) ** 2)
    assert deep_mse < shallow_mse * 0.5
