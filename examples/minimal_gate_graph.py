"""Minimal LangGraph integration of the gate (plan section 4.9).

A two-node graph: an agent node proposes a gated action, the gate node
either passes it through or interrupts for human approval and resumes with
Command(resume={"approved": ...}). Run directly to see the pause/resume
cycle on a canned record:

    python examples/minimal_gate_graph.py
"""

import dataclasses
import uuid

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from gate.middleware import gate_node
from gate.types import StepRecord


def agent_node(state):
    # stand-in for the real agent: proposes one gated write action
    record = StepRecord(
        task_id=0,
        split="demo",
        trial=0,
        step_index=2,
        tool_name="modify_user_address",
        kwargs={"address1": "5 Pine St"},
        tier="write",
        verbalized=0.62,
        agreement=0.6,
    )
    return {"record": dataclasses.asdict(record)}


def build_graph():
    from gate.middleware import GateGraphState
    builder = StateGraph(GateGraphState)
    builder.add_node("agent", agent_node)
    builder.add_node("gate", gate_node)
    builder.add_edge(START, "agent")
    builder.add_edge("agent", "gate")
    builder.add_edge("gate", END)
    return builder.compile(checkpointer=MemorySaver())


def main():
    graph = build_graph()
    config = {"configurable": {"thread_id": uuid.uuid4().hex}}
    state = {"policy_name": "fixed_threshold"}

    result = graph.invoke(state, config)
    # fixed_threshold with verbalized 0.62 < 0.7 -> the graph pauses
    print("interrupt payload:", result["__interrupt__"][0].value)

    resume = graph.invoke(
        Command(resume={"approved": False}), config  # reviewer says no
    )
    print("after resume:", {k: resume[k] for k in ("action", "approved", "gate_reason")})
    assert resume["action"] == "ask" and resume["approved"] is False


if __name__ == "__main__":
    main()
