import random

from gate.threshold import (
    approvals_needed,
    choose_lambda,
    mean_loss,
    trajectory_loss,
)


def _traj(rng, n=2):
    steps, p_hats = [], []
    for _ in range(n):
        p = round(rng.random(), 3)
        correct = rng.random() < p  # labels generated from p_hat
        steps.append({"tier": "write", "correct": correct})
        p_hats.append(p)
    return steps, p_hats


def test_loss_is_non_increasing_in_lambda():
    rng = random.Random(0)
    steps, p_hats = _traj(rng, n=6)
    prev = trajectory_loss(steps, p_hats, 0.0)
    for lam in [i / 20 for i in range(21)]:
        cur = trajectory_loss(steps, p_hats, lam)
        assert cur <= prev
        prev = cur


def test_irreversible_steps_do_not_count_in_loss():
    steps = [
        {"tier": "irreversible", "correct": False},
        {"tier": "write", "correct": False},
    ]
    # the wrong irreversible step never counts (always asks under policy)
    assert trajectory_loss(steps, [0.99, 0.99], 0.5) == 1.0
    assert trajectory_loss(steps, [0.99, 0.99], 0.999) == 0.0


def test_choose_lambda_small_n_unreachable():
    # alpha=0.02 needs N >= 49; with 10 trajectories no lambda can satisfy
    by_traj = {}
    phat = {}
    rng = random.Random(1)
    for t in range(10):
        s, p = _traj(rng)
        by_traj[t], phat[t] = s, p
    assert choose_lambda(by_traj, phat, 0.02) == 1.0


def test_choose_lambda_mc_risk_control():
    """Mandatory test (plan 4.6), stated precisely. Conformal risk control
    guarantees E[L(lambda_hat)] <= alpha in expectation. Two consequences,
    both checked here: (1) the AVERAGE empirical loss on new trajectories
    across repetitions is <= alpha; (2) any single repetition can exceed
    alpha by sampling noise alone (std ~ sqrt(alpha(1-alpha)/n_new)), so
    the fraction of repetitions above alpha + 2 standard errors is small.
    The plan's literal phrasing ("empirical rate <= alpha in the great
    majority of repetitions") only holds with that noise margin; noted in
    notes/progress.md."""
    rng = random.Random(42)

    def draw(n):
        by_traj, phat = {}, {}
        for t in range(n):
            s, p = _traj(rng, n=rng.randrange(1, 4))
            by_traj[t], phat[t] = s, p
        return by_traj, phat

    for alpha in (0.05, 0.10, 0.20):
        losses = []
        reps = 200
        for _ in range(reps):
            cal_s, cal_p = draw(300)
            lam = choose_lambda(cal_s, cal_p, alpha)
            assert lam < 1.0  # reachable with N=300 for these alphas
            new_s, new_p = draw(200)
            losses.append(mean_loss(new_s, new_p, lam))
        avg = sum(losses) / reps
        assert avg <= alpha + 0.005, (alpha, avg)  # expectation bound
        se = (alpha * (1 - alpha) / 200) ** 0.5
        tail = sum(1 for L in losses if L > alpha + 2 * se) / reps
        assert tail <= 0.10, (alpha, tail)


def test_approvals_count():
    steps = [
        {"tier": "write", "correct": True},
        {"tier": "write", "correct": True},
        {"tier": "irreversible", "correct": True},
        {"tier": "write", "correct": True},
    ]
    # p_hats: 0.9, 0.5, 0.9, 0.99 with lam=0.8: asks on step 2 and the
    # irreversible step -> 2 approvals
    assert approvals_needed(steps, [0.9, 0.5, 0.9, 0.99], 0.8) == 2
    assert approvals_needed(steps, [0.9, 0.5, 0.9, 0.99], 0.0) == 1  # only irreversible
