"""Risk-controlled threshold (plan section 4.6).

For each trajectory t and threshold lambda: L_t(lambda) = 1 if some
write-tier step would have been executed without approval (p_hat >= lambda)
and was wrong. L is non-increasing in lambda. The conformal rule picks the
smallest lambda such that (N/(N+1)) * mean(L_t) + 1/(N+1) <= alpha.

Unit: the trajectory, not the step. Limitation (documented in the README):
calibration trajectories come from ungated runs; in deployment the gate
changes the path after a rejection, so the guarantee is an approximation.
"""

from typing import Dict, List, Sequence

ALPHAS = (0.02, 0.05, 0.10, 0.20)


def trajectory_loss(steps: Sequence[Dict], p_hats: Sequence[float], lam: float) -> float:
    loss = 0.0
    for step, p in zip(steps, p_hats):
        if step["tier"] == "write" and p >= lam and not step["correct"]:
            loss = 1.0
            break
    return loss


def mean_loss(steps_by_traj: Dict[int, Sequence[Dict]],
              p_hat_by_traj: Dict[int, Sequence[float]], lam: float) -> float:
    losses = [
        trajectory_loss(steps_by_traj[t], p_hat_by_traj[t], lam)
        for t in steps_by_traj
    ]
    return sum(losses) / len(losses) if losses else 1.0


def choose_lambda(steps_by_traj, p_hat_by_traj, alpha: float) -> float:
    """Smallest lambda on the observed p_hat grid satisfying the bound.

    Returns 1.0 (ask always) if no lambda satisfies it, including the case
    N < 1/alpha - 1 where the bound is unreachable for any lambda.
    """
    N = len(steps_by_traj)
    if N == 0:
        return 1.0
    grid = sorted({1.0} | {p for ps in p_hat_by_traj.values() for p in ps})
    for lam in grid:
        bound = (N / (N + 1)) * mean_loss(steps_by_traj, p_hat_by_traj, lam) + 1 / (N + 1)
        if bound <= alpha:
            return lam
    return 1.0


def approvals_needed(steps: Sequence[Dict], p_hats: Sequence[float], lam: float) -> int:
    """Approvals a calibrated policy would have requested on a trajectory:
    every write step with p_hat < lambda, plus every irreversible step."""
    n = 0
    for step, p in zip(steps, p_hats):
        if step["tier"] == "irreversible" or (step["tier"] == "write" and p < lam):
            n += 1
    return n


def expected_approvals(steps_by_traj, p_hat_by_traj, lam: float) -> float:
    vals = [
        approvals_needed(steps_by_traj[t], p_hat_by_traj[t], lam)
        for t in steps_by_traj
    ]
    return sum(vals) / len(vals) if vals else 0.0
