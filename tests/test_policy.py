from gate.policy import (
    CalibratedPolicy,
    ConfirmAllPolicy,
    FixedThresholdPolicy,
    make_policy,
)
from gate.types import StepRecord


def _record(tier="write", verbalized=None, p_hat=None):
    return StepRecord(
        task_id=0,
        split="test",
        trial=0,
        step_index=0,
        tool_name="modify_user_address",
        kwargs={},
        tier=tier,
        verbalized=verbalized,
        p_hat=p_hat,
    )


def test_confirm_all_asks_every_gated_action():
    p = ConfirmAllPolicy()
    assert p.decide(_record("write")).action == "ask"
    assert p.decide(_record("irreversible")).action == "ask"
    assert p.decide(_record("read")).action == "act"


def test_fixed_threshold_uses_raw_verbalized():
    p = FixedThresholdPolicy()
    assert p.decide(_record("write", verbalized=0.9)).action == "act"
    assert p.decide(_record("write", verbalized=0.5)).action == "ask"
    assert p.decide(_record("write", verbalized=None)).action == "ask"
    assert p.decide(_record("irreversible", verbalized=0.99)).action == "ask"


def test_calibrated_uses_p_hat_and_lambda():
    p = CalibratedPolicy(lam=0.8)
    assert p.decide(_record("write", p_hat=0.9)).action == "act"
    assert p.decide(_record("write", p_hat=0.7)).action == "ask"
    assert p.decide(_record("write", p_hat=None)).action == "ask"
    assert p.decide(_record("irreversible", p_hat=0.99)).action == "ask"


def test_make_policy():
    assert make_policy("confirm_all").name == "confirm_all"
    assert make_policy("calibrated", lam=0.5).lam == 0.5
    import pytest

    with pytest.raises(ValueError):
        make_policy("calibrated")
    with pytest.raises(ValueError):
        make_policy("nope")
