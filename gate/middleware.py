"""Instrumented agent loop: the gate hook of the plan (section 4.9).

`InstrumentedAgent.solve` is the tau-bench `ToolCallingAgent.solve` loop
(vendor/tau-bench, commit 59a200c) with one hook: between
`message_to_action` and `env.step`, every gated (write / irreversible)
action goes through `self.gate(record) -> GateDecision` after its label
(and signals, if the policy needs them) are computed.

- CollectAgent (Phase 2) always acts and records signals: base behavior.
- TauGateAgent (Phase 5) applies a policy; "ask" goes to an oracle that
  approves correct actions and rejects wrong ones with the message
  "action rejected by reviewer", after which the agent replans.

The LangGraph node (gate_node) at the bottom of this file wraps the same
decision in interrupt()/Command(resume=...).
"""

from typing import Any, Dict, List, Optional, Tuple

from litellm import completion
from tau_bench.agents.tool_calling_agent import (
    ToolCallingAgent,
    message_to_action,
)
from tau_bench.types import SolveResult, RESPOND_ACTION_NAME

from gate.labeling import is_correct
from gate.oracle import Oracle
from gate.policy import ConfirmAllPolicy
from gate.signals import agreement, verbalized_confidence
from gate.tiers import gated, tier_of
from gate.types import GateDecision, StepRecord


class InstrumentedAgent(ToolCallingAgent):
    def __init__(self, *args, calibrator=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.calibrator = calibrator
        self.stats = {
            "approvals_requested": 0,
            "executed_wrong_writes": 0,
            "rejections": 0,
        }

    # subclasses override
    needs_signals = True

    def gate(self, record: StepRecord) -> GateDecision:
        raise NotImplementedError

    def oracle(self, record: StepRecord) -> bool:
        return True

    def solve(
        self, env, task_index: Optional[int] = None, max_num_steps: int = 30
    ) -> Tuple[SolveResult, List[StepRecord]]:
        records: List[StepRecord] = []
        env_reset_res = env.reset(task_index=task_index)
        obs = env_reset_res.observation
        info = env_reset_res.info.model_dump()
        reward = 0.0
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": self.wiki},
            {"role": "user", "content": obs},
        ]
        gold = env.task.actions
        task_id = env.task_index
        step = 0
        for _ in range(max_num_steps):
            res = completion(
                messages=messages,
                model=self.model,
                custom_llm_provider=self.provider,
                tools=self.tools_info,
                temperature=self.temperature,
            )
            next_message = res.choices[0].message.model_dump()
            action = message_to_action(next_message)
            try:
                is_gated = action.name != RESPOND_ACTION_NAME and gated(action.name)
            except KeyError:
                is_gated = False  # hallucinated tool: env.step returns an
                # error observation, exactly like the base loop
            rejected = False
            if is_gated:
                record = StepRecord(
                    task_id=task_id,
                    split=getattr(self, "split_label", "?"),
                    trial=getattr(self, "trial", 0),
                    step_index=step,
                    tool_name=action.name,
                    kwargs=dict(action.kwargs),
                    tier=tier_of(action.name),
                )
                record.correct = is_correct(action.name, action.kwargs, gold)
                if self.needs_signals:
                    record.verbalized, v_usage = verbalized_confidence(
                        messages, action.name, action.kwargs, model=self.model
                    )
                    record.agreement, a_usage = agreement(
                        messages,
                        self.tools_info,
                        action.name,
                        action.kwargs,
                        model=self.model,
                    )
                    record.signal_usage = {
                        "verbalized": v_usage,
                        "agreement": a_usage,
                    }
                    if self.calibrator is not None:
                        import dataclasses as _dc

                        record.p_hat = float(
                            self.calibrator.predict_proba([_dc.asdict(record)])[0]
                        )
                records.append(record)
                decision = self.gate(record)
                record.gate = {
                    "policy": decision.policy,
                    "action": decision.action,
                    "reason": decision.reason,
                }
                if decision.action == "ask":
                    approved = self.oracle(record)
                    record.gate["approved"] = approved
                    if not approved:
                        self.stats["rejections"] += 1
                        rejected = True
                if not rejected:
                    if (
                        decision.action == "act"
                        and record.tier == "write"
                        and not record.correct
                    ):
                        self.stats["executed_wrong_writes"] += 1
                    env_response = env.step(action)
                    reward = env_response.reward
                    info = {**info, **env_response.info.model_dump()}
            else:
                env_response = env.step(action)
                reward = env_response.reward
                info = {**info, **env_response.info.model_dump()}

            if rejected:
                messages.extend(
                    [
                        next_message,
                        {
                            "role": "tool",
                            "tool_call_id": next_message["tool_calls"][0]["id"],
                            "name": next_message["tool_calls"][0]["function"]["name"],
                            "content": "action rejected by reviewer",
                        },
                    ]
                )
            elif action.name != RESPOND_ACTION_NAME:
                next_message["tool_calls"] = next_message["tool_calls"][:1]
                messages.extend(
                    [
                        next_message,
                        {
                            "role": "tool",
                            "tool_call_id": next_message["tool_calls"][0]["id"],
                            "name": next_message["tool_calls"][0]["function"]["name"],
                            "content": env_response.observation,
                        },
                    ]
                )
            else:
                messages.extend(
                    [
                        next_message,
                        {"role": "user", "content": env_response.observation},
                    ]
                )
            step += 1
            if not rejected and env_response.done:
                break
        return (
            SolveResult(
                reward=reward,
                info=info,
                messages=messages,
                total_cost=0.0,  # real accounting lives in gate.llm.tracker
            ),
            records,
        )


class CollectAgent(InstrumentedAgent):
    """Base agent with signal collection: everything executes, nothing asks."""

    needs_signals = True

    def gate(self, record: StepRecord) -> GateDecision:
        return GateDecision(policy="collect", action="act", reason="collect")


class TauGateAgent(InstrumentedAgent):
    """Policy-gated agent with the perfect oracle of plan 4.8.

    calibrator: optional fitted calibrator (variant A or B). When set and
    the policy needs signals, record.p_hat is computed before deciding, so
    the calibrated policy has its input.
    """

    def __init__(self, *args, policy=None, oracle=None, calibrator=None, **kwargs):
        super().__init__(*args, calibrator=calibrator, **kwargs)
        self.policy = policy or ConfirmAllPolicy()
        self._oracle = oracle
        self.needs_signals = getattr(self.policy, "needs_signals", True)

    def gate(self, record: StepRecord) -> GateDecision:
        return self.policy.decide(record)

    def oracle(self, record: StepRecord) -> bool:
        self.stats["approvals_requested"] += 1
        if self._oracle is None:  # default: perfect oracle on the label
            return bool(record.correct)
        return self._oracle.approve(record)


from typing import TypedDict  # noqa: E402  (used by the graph state below)


class GateGraphState(TypedDict, total=False):
    record: Dict[str, Any]  # StepRecord as dict
    policy_name: str  # "confirm_all" | "fixed_threshold" | "calibrated"
    lam: float  # lambda for the calibrated policy
    approved: bool
    action: str
    gate_reason: str


def gate_node(state: "GateGraphState") -> Dict[str, Any]:
    """LangGraph node wrapping the gate decision in interrupt().

    Use StateGraph(GateGraphState): a plain `dict` schema silently drops
    keys between nodes. On resume, expects Command(resume={"approved":
    bool}). Integrators call this node between the agent step and the tool
    execution node.
    """
    from langgraph.types import interrupt

    from gate.policy import make_policy
    from gate.types import StepRecord

    record = (
        state["record"]
        if isinstance(state["record"], StepRecord)
        else StepRecord(**state["record"])
    )
    policy = make_policy(state.get("policy_name") or "confirm_all", state.get("lam"))
    decision = policy.decide(record)
    if decision.action == "act":
        return {"approved": True, "action": "act", "gate_reason": decision.reason}
    answer = interrupt(
        {
            "question": "approve this action?",
            "record": {
                "tool": record.tool_name,
                "kwargs": record.kwargs,
                "tier": record.tier,
                "p_hat": record.p_hat,
                "verbalized": record.verbalized,
                "agreement": record.agreement,
            },
            "gate_reason": decision.reason,
        }
    )
    approved = bool(answer.get("approved"))
    return {
        "approved": approved,
        "action": "ask",
        "gate_reason": decision.reason,
    }
