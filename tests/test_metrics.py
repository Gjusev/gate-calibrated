import math

from eval.metrics import brier, ece, pass_hat_k, reliability


def test_ece_zero_when_perfectly_calibrated():
    import random

    rng = random.Random(0)
    ys, ps = [], []
    for i in range(1, 101):
        p = i / 100
        for _ in range(6):
            ps.append(p)
            ys.append(1.0 if rng.random() < p else 0.0)
    e = ece(ys, ps)
    assert e < 0.05  # sampling noise only; deterministic with the seed


def test_ece_large_when_overconfident():
    ys = [0.0] * 50 + [1.0] * 50
    ps = [0.95] * 100  # confident, half wrong
    assert ece(ys, ps) > 0.4


def test_brier_value():
    assert brier([1.0, 0.0], [0.8, 0.3]) == pytest_approx((0.04 + 0.09) / 2)


def pytest_approx(x):
    import pytest

    return pytest.approx(x)


def test_pass_hat_k():
    rewards = {1: [1, 1, 0, 0], 2: [1, 1, 1, 1]}
    # task 1: c=2,n=4 -> pass^1 = 0.5, pass^2 = C(2,2)/C(4,2) = 1/6
    # task 2: all success -> 1.0
    assert pass_hat_k(rewards, 1) == pytest_approx(0.75)
    assert pass_hat_k(rewards, 2) == pytest_approx((1 / 6 + 1.0) / 2)
    assert pass_hat_k(rewards, 4) == pytest_approx(0.5)  # 0.0 and 1.0
    assert pass_hat_k(rewards, 5) == 0.0  # k > n everywhere


def test_reliability_equal_mass():
    ps = [i / 10 for i in range(10)]
    ys = [1 if p >= 0.5 else 0 for p in ps]
    rel = reliability(ys, ps, n_bins=5)
    assert sum(r["count"] for r in rel) == 10
    assert all(r["count"] == 2 for r in rel)
