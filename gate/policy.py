"""Policies (plan section 4.7). decide(record) -> GateDecision.

confirm_all: ask on every write and irreversible action.
fixed_threshold: ask if verbalized < 0.7 (a hand-picked tutorial value);
    irreversible always asks.
calibrated: ask if p_hat < lambda or irreversible.
"""

from typing import Optional

from gate.types import GateDecision, StepRecord

FIXED_THRESHOLD = 0.7


class ConfirmAllPolicy:
    name = "confirm_all"
    # asks on every gated action: no signals needed, skip their cost
    needs_signals = False

    def decide(self, record: StepRecord) -> GateDecision:
        if record.tier in ("write", "irreversible"):
            return GateDecision(policy=self.name, action="ask",
                                reason="ask on every gated action")
        return GateDecision(policy=self.name, action="act", reason="not gated")


class FixedThresholdPolicy:
    name = "fixed_threshold"
    needs_signals = True  # reads record.verbalized

    def __init__(self, threshold: float = FIXED_THRESHOLD):
        self.threshold = threshold

    def decide(self, record: StepRecord) -> GateDecision:
        if record.tier == "irreversible":
            return GateDecision(policy=self.name, action="ask",
                                reason="irreversible always asks")
        if record.tier == "write":
            v = record.verbalized
            ask = v is None or v < self.threshold
            return GateDecision(
                policy=self.name,
                action="ask" if ask else "act",
                p_hat=v,
                reason=f"verbalized={v} {'<' if ask else '>='} {self.threshold}",
            )
        return GateDecision(policy=self.name, action="act", reason="not gated")


class CalibratedPolicy:
    name = "calibrated"
    needs_signals = True  # reads record.p_hat

    def __init__(self, lam: float):
        self.lam = lam

    def decide(self, record: StepRecord) -> GateDecision:
        if record.tier == "irreversible":
            return GateDecision(policy=self.name, action="ask",
                                reason="irreversible always asks")
        if record.tier == "write":
            p = record.p_hat
            ask = p is None or p < self.lam
            return GateDecision(
                policy=self.name,
                action="ask" if ask else "act",
                p_hat=p,
                reason=f"p_hat={p} {'<' if ask else '>='} lambda={self.lam}",
            )
        return GateDecision(policy=self.name, action="act", reason="not gated")


def make_policy(name: str, lam: Optional[float] = None):
    if name == "confirm_all":
        return ConfirmAllPolicy()
    if name == "fixed_threshold":
        return FixedThresholdPolicy()
    if name == "calibrated":
        if lam is None:
            raise ValueError("calibrated policy needs lambda")
        return CalibratedPolicy(lam)
    raise ValueError(f"unknown policy {name!r}")
