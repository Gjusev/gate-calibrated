"""Test of the LangGraph interrupt/resume cycle (plan section 4.9)."""

import dataclasses
import uuid

from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from gate.middleware import gate_node
from gate.types import StepRecord


def _record(**over):
    base = dict(
        task_id=0,
        split="demo",
        trial=0,
        step_index=1,
        tool_name="modify_user_address",
        kwargs={"address1": "5 Pine St"},
        tier="write",
        verbalized=0.5,
        agreement=0.6,
    )
    base.update(over)
    return StepRecord(**base)


def _graph():
    from langgraph.graph import END, START, StateGraph

    from gate.middleware import GateGraphState
    builder = StateGraph(GateGraphState)
    builder.add_node("gate", gate_node)
    builder.add_edge(START, "gate")
    builder.add_edge("gate", END)
    return builder.compile(checkpointer=MemorySaver())


def test_act_decision_does_not_interrupt():
    graph = _graph()
    config = {"configurable": {"thread_id": uuid.uuid4().hex}}
    state = {
        "record": dataclasses.asdict(_record(verbalized=0.95)),
        "policy_name": "fixed_threshold",
    }
    result = graph.invoke(state, config)
    assert result["action"] == "act"
    assert result["approved"] is True
    assert "__interrupt__" not in result


def test_ask_decision_interrupts_and_resumes():
    graph = _graph()
    config = {"configurable": {"thread_id": uuid.uuid4().hex}}
    state = {
        "record": dataclasses.asdict(_record(verbalized=0.4)),
        "policy_name": "fixed_threshold",
    }
    result = graph.invoke(state, config)
    assert "__interrupt__" in result
    payload = result["__interrupt__"][0].value
    assert payload["record"]["tool"] == "modify_user_address"

    resumed = graph.invoke(Command(resume={"approved": True}), config)
    assert resumed["action"] == "ask"
    assert resumed["approved"] is True

    # a second thread, reviewer says no
    config2 = {"configurable": {"thread_id": uuid.uuid4().hex}}
    graph.invoke(state, config2)
    resumed2 = graph.invoke(Command(resume={"approved": False}), config2)
    assert resumed2["approved"] is False


def test_calibrated_policy_in_node():
    graph = _graph()
    config = {"configurable": {"thread_id": uuid.uuid4().hex}}
    # p_hat 0.9 >= lambda 0.8 -> act without interruption
    state = {
        "record": dataclasses.asdict(_record(p_hat=0.9)),
        "policy_name": "calibrated",
        "lam": 0.8,
    }
    result = graph.invoke(state, config)
    assert result["action"] == "act"
