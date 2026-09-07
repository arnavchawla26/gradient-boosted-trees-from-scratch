import numpy as np
import pytest

from gbdt.losses import LeastSquaresLoss, LogisticLoss, get_loss


def _finite_diff_grad(loss, y, raw_pred, eps=1e-6):
    grad = np.empty_like(raw_pred)
    for i in range(len(raw_pred)):
        up = raw_pred.copy()
        down = raw_pred.copy()
        up[i] += eps
        down[i] -= eps
        # loss() returns the mean over all samples, so undo the averaging
        # to recover the per-sample derivative.
        n = len(raw_pred)
        grad[i] = (loss.loss(y, up) - loss.loss(y, down)) / (2 * eps) * n
    return grad


def _finite_diff_hess(loss, y, raw_pred, eps=1e-4):
    hess = np.empty_like(raw_pred)
    n = len(raw_pred)
    for i in range(len(raw_pred)):
        up = raw_pred.copy()
        down = raw_pred.copy()
        up[i] += eps
        down[i] -= eps
        g_up = (loss.loss(y, up) - loss.loss(y, raw_pred)) / eps * n
        g_down = (loss.loss(y, raw_pred) - loss.loss(y, down)) / eps * n
        hess[i] = (g_up - g_down) / eps
    return hess


class TestLeastSquaresLoss:
    def test_init_score_is_mean(self):
        y = np.array([1.0, 2.0, 3.0, 10.0])
        assert LeastSquaresLoss().init_score(y) == pytest.approx(np.mean(y))

    def test_gradient_matches_finite_difference(self):
        rng = np.random.default_rng(0)
        y = rng.normal(size=20)
        raw_pred = rng.normal(size=20)
        loss = LeastSquaresLoss()
        analytic = loss.gradient(y, raw_pred)
        numeric = _finite_diff_grad(loss, y, raw_pred)
        np.testing.assert_allclose(analytic, numeric, atol=1e-4)

    def test_hessian_is_constant_one(self):
        y = np.array([1.0, 5.0, -3.0])
        raw_pred = np.array([0.5, 0.5, 0.5])
        hess = LeastSquaresLoss().hessian(y, raw_pred)
        np.testing.assert_array_equal(hess, np.ones(3))

    def test_loss_zero_at_perfect_prediction(self):
        y = np.array([1.0, 2.0, 3.0])
        assert LeastSquaresLoss().loss(y, y) == 0.0

    def test_link_is_identity(self):
        raw = np.array([1.0, -2.0, 3.5])
        np.testing.assert_array_equal(LeastSquaresLoss().link(raw), raw)


class TestLogisticLoss:
    def test_init_score_is_logit_of_mean(self):
        y = np.array([1.0, 1.0, 0.0, 0.0])  # mean = 0.5
        assert LogisticLoss().init_score(y) == pytest.approx(0.0, abs=1e-9)

    def test_init_score_matches_class_imbalance(self):
        y = np.array([1.0] * 9 + [0.0])  # mean = 0.9
        score = LogisticLoss().init_score(y)
        p = 1 / (1 + np.exp(-score))
        assert p == pytest.approx(0.9, abs=1e-6)

    def test_gradient_matches_finite_difference(self):
        rng = np.random.default_rng(1)
        y = rng.integers(0, 2, size=20).astype(np.float64)
        raw_pred = rng.normal(size=20)
        loss = LogisticLoss()
        analytic = loss.gradient(y, raw_pred)
        numeric = _finite_diff_grad(loss, y, raw_pred)
        np.testing.assert_allclose(analytic, numeric, atol=1e-3)

    def test_hessian_matches_finite_difference(self):
        rng = np.random.default_rng(2)
        y = rng.integers(0, 2, size=15).astype(np.float64)
        raw_pred = rng.normal(scale=0.5, size=15)
        loss = LogisticLoss()
        analytic = loss.hessian(y, raw_pred)
        numeric = _finite_diff_hess(loss, y, raw_pred)
        np.testing.assert_allclose(analytic, numeric, atol=1e-2)

    def test_gradient_sign_matches_error_direction(self):
        # If the model predicts "confidently positive" (large raw score)
        # for a true negative, gradient (p - y) should be strongly positive
        # (pushing the next tree to lower the raw score).
        y = np.array([0.0])
        raw_pred = np.array([5.0])
        grad = LogisticLoss().gradient(y, raw_pred)
        assert grad[0] > 0.9

    def test_hessian_is_bounded_between_0_and_quarter(self):
        rng = np.random.default_rng(3)
        y = rng.integers(0, 2, size=50).astype(np.float64)
        raw_pred = rng.normal(scale=3, size=50)
        hess = LogisticLoss().hessian(y, raw_pred)
        assert np.all(hess > 0)
        assert np.all(hess <= 0.25 + 1e-9)

    def test_link_is_sigmoid_and_bounded(self):
        raw = np.array([-100.0, 0.0, 100.0])
        p = LogisticLoss().link(raw)
        assert p[0] == pytest.approx(0.0, abs=1e-9)
        assert p[1] == pytest.approx(0.5)
        assert p[2] == pytest.approx(1.0, abs=1e-9)

    def test_loss_decreases_as_prediction_improves(self):
        y = np.array([1.0])
        loss = LogisticLoss()
        worse = loss.loss(y, np.array([-5.0]))
        better = loss.loss(y, np.array([5.0]))
        assert better < worse


def test_get_loss_returns_correct_type():
    assert isinstance(get_loss("squared_error"), LeastSquaresLoss)
    assert isinstance(get_loss("log_loss"), LogisticLoss)


def test_get_loss_rejects_unknown_name():
    with pytest.raises(ValueError):
        get_loss("not_a_real_loss")
