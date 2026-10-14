"""Phase 5 verification tests: stub model proposes a wrong action and a
correct one; the oracle rejects the first and approves the second
(plan section 6, Phase 5)."""

import pytest
from tau_bench.envs.retail.env import MockRetailDomainEnv

from gate.middleware import TauGateAgent
from gate.policy import ConfirmAllPolicy, FixedThresholdPolicy

GOLD = {
    "order_id": "#W6067464",
    "item_ids": ["9624127908", "8917609800"],
    "payment_method_id": "credit_card_4190576",
}
WRONG = dict(GOLD, item_ids=["1111111111"])


def _fake(turns, make_response, verbalized="0.99", n_matching=5):
    state = {"i": 0, "samples": 0}

    def _is_assessor(kwargs):
        tools = kwargs.get("tools") or []
        return bool(tools) and tools[0]["function"]["name"] == "report_probability"

    def fake(**kwargs):
        if kwargs.get("tools") and kwargs.get("temperature") == 0.7:
            # agreement sample: repeat the most recent agent turn (the
            # current proposal)
            cur = turns[max(state["i"] - 1, 0)]
            return make_response(tool_call=cur)
        if _is_assessor(kwargs):
            return make_response(content=verbalized)
        name, kw = turns[state["i"]]
        state["i"] += 1
        return make_response(tool_call=(name, kw))

    return fake


def _setup(monkeypatch, fake):
    import gate.middleware as mw
    import gate.signals as sig

    monkeypatch.setattr(mw, "completion", fake)
    monkeypatch.setattr(sig, "completion", fake)
    monkeypatch.setattr("builtins.input", lambda prompt="": "hello")


def test_oracle_rejects_wrong_and_approves_correct(monkeypatch, make_response):
    # agent proposes: read, WRONG write (rejected -> replan), GOLD write
    # (approved -> executed), done
    turns = [
        ("get_order_details", {"order_id": "#W6067464", "user_id": "omar_anderson_3203"}),
        ("return_delivered_order_items", WRONG),
        ("return_delivered_order_items", GOLD),
        ("think", {"thought": "done"}),
    ]
    _setup(monkeypatch, _fake(turns, make_response))

    env = MockRetailDomainEnv(user_strategy="human", task_split="train", task_index=0)
    agent = TauGateAgent(
        tools_info=env.tools_info,
        wiki=env.wiki,
        model="f",
        provider="x",
        policy=ConfirmAllPolicy(),
    )
    result, records = agent.solve(env, task_index=0, max_num_steps=4)

    assert agent.stats["approvals_requested"] == 2
    assert agent.stats["rejections"] == 1
    assert agent.stats["executed_wrong_writes"] == 0
    # the wrong action never reached the environment
    executed_returns = [
        a for a in env.actions if a.name == "return_delivered_order_items"
    ]
    assert len(executed_returns) == 1
    from gate.labeling import canonical

    assert canonical(executed_returns[0].kwargs, "return_delivered_order_items") == (
        canonical(GOLD, "return_delivered_order_items")
    )
    # records: 2 gated steps (wrong rejected, correct approved)
    assert [r.gate["approved"] for r in records] == [False, True]


def test_fixed_threshold_executes_wrong_when_confident(monkeypatch, make_response):
    # verbalized 0.99 >= 0.7 -> act without ask; the wrong WRITE executes.
    # (A write-tier tool: irreversible ones always ask, and return_delivered
    # in the other test is irreversible on purpose.)
    wrong_write = {
        "address1": "5 Wrong St",
        "city": "X",
        "state": "CA",
        "country": "US",
        "zip": "90001",
    }
    turns = [
        ("get_order_details", {"order_id": "#W6067464", "user_id": "omar_anderson_3203"}),
        ("modify_user_address", wrong_write),  # write tier, not in gold
        ("think", {"thought": "done"}),
    ]
    _setup(monkeypatch, _fake(turns, make_response, verbalized="0.99"))

    env = MockRetailDomainEnv(user_strategy="human", task_split="train", task_index=0)
    agent = TauGateAgent(
        tools_info=env.tools_info,
        wiki=env.wiki,
        model="f",
        provider="x",
        policy=FixedThresholdPolicy(),
    )
    result, records = agent.solve(env, task_index=0, max_num_steps=3)

    assert agent.stats["approvals_requested"] == 0
    assert agent.stats["executed_wrong_writes"] == 1
    assert records[0].gate["action"] == "act"
