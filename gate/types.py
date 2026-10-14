"""Dataclasses shared by the gate modules."""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class StepRecord:
    """One proposed action of the agent, with its tier, signals and label.

    Signals and label are filled in by signals.py / labeling.py; a StepRecord
    can exist before they are computed (tiers and actions are known upfront).
    """

    task_id: int
    split: str
    trial: int
    step_index: int  # number of previous agent steps in this trajectory
    tool_name: str
    kwargs: Dict[str, Any]
    tier: str  # "read" | "write" | "irreversible" | "special"
    verbalized: Optional[float] = None
    agreement: Optional[float] = None
    correct: Optional[bool] = None
    p_hat: Optional[float] = None
    # token usage of the signal calls, set by middleware during collection
    signal_usage: Optional[Dict[str, Any]] = field(default=None)
    # gate outcome for this step: policy, action, reason, approved
    gate: Optional[Dict[str, Any]] = field(default=None)


@dataclass
class GateDecision:
    """The decision of a policy for one StepRecord."""

    policy: str
    action: str  # "act" | "ask"
    p_hat: Optional[float] = None
    reason: str = ""
