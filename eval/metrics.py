"""Evaluation metrics (plan sections 4.5, 4.8): ECE (10 equal-mass bins),
Brier, reliability diagram, pass^k with the unbiased estimator, and the
per-trajectory risk loss for thresholds."""

from collections import defaultdict
from math import comb
from typing import Dict, List, Sequence


def ece(y_true: Sequence[float], p_pred: Sequence[float], n_bins: int = 10) -> float:
    """Expected calibration error with equal-mass bins."""
    order = sorted(range(len(p_pred)), key=lambda i: p_pred[i])
    n = len(p_pred)
    total = 0.0
    for b in range(n_bins):
        idx = order[b * n // n_bins : (b + 1) * n // n_bins]
        if not idx:
            continue
        conf = sum(p_pred[i] for i in idx) / len(idx)
        acc = sum(y_true[i] for i in idx) / len(idx)
        total += (len(idx) / n) * abs(conf - acc)
    return total


def brier(y_true: Sequence[float], p_pred: Sequence[float]) -> float:
    return sum((p - y) ** 2 for y, p in zip(y_true, p_pred)) / len(y_true)


def reliability(
    y_true: Sequence[float], p_pred: Sequence[float], n_bins: int = 10
) -> List[Dict]:
    """Equal-mass bins with accuracy, mean confidence and count."""
    order = sorted(range(len(p_pred)), key=lambda i: p_pred[i])
    n = len(p_pred)
    out = []
    for b in range(n_bins):
        idx = order[b * n // n_bins : (b + 1) * n // n_bins]
        if not idx:
            continue
        out.append(
            {
                "bin": b,
                "count": len(idx),
                "accuracy": sum(y_true[i] for i in idx) / len(idx),
                "confidence": sum(p_pred[i] for i in idx) / len(idx),
            }
        )
    return out


def pass_hat_k(rewards_per_task: Dict[int, List[float]], k: int) -> float:
    """Unbiased pass^k estimator: mean over tasks of C(c, k)/C(n, k),
    where c is the number of successful trials and n the number run."""
    vals = []
    for task_id, rewards in rewards_per_task.items():
        n = len(rewards)
        c = sum(1 for r in rewards if r >= 1.0)
        if n == 0:
            continue
        if k > n:
            vals.append(0.0)  # not enough trials: cannot draw k successes
        else:
            vals.append(comb(c, k) / comb(n, k))
    return sum(vals) / len(vals) if vals else 0.0


def trajectory_losses(
    steps_by_trajectory: Dict[int, Sequence[Dict]], p_hat_by_step, lam: float
) -> List[float]:
    """L_t(lambda) = 1 if any write-tier step with p_hat >= lambda was wrong.

    Only write tier counts (irreversible actions always ask, so they are
    never executed without approval under any policy).
    p_hat_by_step: mapping trajectory_id -> list of p_hat aligned with the
    steps sequence.
    """
    losses = []
    for tid, steps in steps_by_trajectory.items():
        p_hats = p_hat_by_step[tid]
        loss = 0.0
        for step, p in zip(steps, p_hats):
            if step["tier"] == "write" and p >= lam and not step["correct"]:
                loss = 1.0
                break
        losses.append(loss)
    return losses
