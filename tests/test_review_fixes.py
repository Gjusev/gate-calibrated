"""Regression tests for the findings of the adversarial review round
(2026-10-13). One test per confirmed defect."""

import json

import litellm
import pytest

from gate.labeling import canonical
from gate.llm import install, uninstall


def test_exchange_lists_sort_independently():
    # exchange stores sorted(item_ids)/sorted(new_item_ids); the pairing
    # never reaches the DB, so any pairing with the same elements is the
    # same action
    a = {
        "order_id": "#1",
        "item_ids": ["1", "2"],
        "new_item_ids": ["x", "y"],
        "payment_method_id": "cc",
    }
    b = {
        "order_id": "#1",
        "item_ids": ["2", "1"],
        "new_item_ids": ["y", "x"],
        "payment_method_id": "cc",
    }
    c = {
        "order_id": "#1",
        "item_ids": ["1", "2"],
        "new_item_ids": ["y", "x"],
        "payment_method_id": "cc",
    }
    assert canonical(a, "exchange_delivered_order_items") == canonical(
        b, "exchange_delivered_order_items"
    )
    # different pairing, same elements: also the same action for exchange
    # (the pairing never reaches the DB)
    assert canonical(a, "exchange_delivered_order_items") == canonical(
        c, "exchange_delivered_order_items"
    )
    # different elements are still different
    d = {
        "order_id": "#1",
        "item_ids": ["1", "2"],
        "new_item_ids": ["z", "y"],
        "payment_method_id": "cc",
    }
    assert canonical(a, "exchange_delivered_order_items") != canonical(
        d, "exchange_delivered_order_items"
    )


def test_modify_with_duplicate_ids_keeps_pair_order():
    # modify mutates the FIRST matching item per pair; with duplicates the
    # pair order changes the final DB state
    a = {"item_ids": ["1", "1"], "new_item_ids": ["x", "y"]}
    b = {"item_ids": ["1", "1"], "new_item_ids": ["y", "x"]}
    assert canonical(a, "modify_pending_order_items") != canonical(
        b, "modify_pending_order_items"
    )
    # unique ids: consistent reorder is still the same action
    c = {"item_ids": ["1", "2"], "new_item_ids": ["x", "y"]}
    d = {"item_ids": ["2", "1"], "new_item_ids": ["y", "x"]}
    assert canonical(c, "modify_pending_order_items") == canonical(
        d, "modify_pending_order_items"
    )


def test_hallucinated_tool_name_does_not_crash_loop(monkeypatch, make_response):
    from tau_bench.envs.retail.env import MockRetailDomainEnv

    from gate.middleware import CollectAgent

    turns = [
        ("get_order_details", {"order_id": "#W6067464", "user_id": "omar_anderson_3203"}),
        ("modify_user_address", {"address1": "5 Pine St", "city": "X", "state": "CA", "country": "US", "zip": "90001"}),  # gated
        ("this_tool_does_not_exist", {"foo": "bar"}),  # hallucinated: base loop survives
        ("think", {"thought": "end"}),
    ]
    state = {"i": 0}

    def _is_assessor(kwargs):
        tools = kwargs.get("tools") or []
        return bool(tools) and tools[0]["function"]["name"] == "report_probability"

    def fake(**kwargs):
        if kwargs.get("tools") and kwargs.get("temperature") == 0.7:
            name, kw = turns[1]
            return make_response(tool_call=(name, kw))
        if _is_assessor(kwargs):
            return make_response(content="0.9")
        name, kw = turns[state["i"]]
        state["i"] += 1
        return make_response(tool_call=(name, kw))

    import gate.middleware as mw
    import gate.signals as sig

    monkeypatch.setattr(mw, "completion", fake)
    monkeypatch.setattr(sig, "completion", fake)
    monkeypatch.setattr("builtins.input", lambda prompt="": "hello")

    env = MockRetailDomainEnv(user_strategy="human", task_split="train", task_index=0)
    agent = CollectAgent(tools_info=env.tools_info, wiki=env.wiki, model="f", provider="x")
    result, records = agent.solve(env, task_index=0, max_num_steps=4)
    # loop survived the hallucinated name exactly like the base agent:
    # env.step returned an error observation, conversation continued
    assert any(a.name == "this_tool_does_not_exist" for a in env.actions)
    assert len(records) == 1  # the gated modify was still recorded


def test_verbalized_reads_structured_tool_call(monkeypatch, make_response):
    import gate.signals as sig

    seen = {}

    def fake(**kwargs):
        seen.update(kwargs)
        return make_response(
            tool_call=("report_probability", {"probability": 0.37})
        )

    monkeypatch.setattr(sig, "completion", fake)
    value, usage = sig.verbalized_confidence([{"role": "user", "content": "hi"}], "t", {})
    assert value == pytest.approx(0.37)
    assert seen["tools"][0]["function"]["name"] == "report_probability"


def test_verbalized_falls_back_to_last_number(monkeypatch, make_response):
    import gate.signals as sig

    def fake(**kwargs):
        return make_response(
            content="On a scale from 0 to 1, I would say 0.8"
        )

    monkeypatch.setattr(sig, "completion", fake)
    value, _ = sig.verbalized_confidence([{"role": "user", "content": "hi"}], "t", {})
    assert value == pytest.approx(0.8)


def test_parse_probability_last_number_wins():
    from gate.signals import parse_probability

    assert parse_probability("0 to 1 scale: 0.8") == pytest.approx(0.8)
    assert parse_probability("between 0 and 1: I estimate 95%") == pytest.approx(0.95)


def test_done_keys_counts_traj_markers(tmp_path):
    from eval.collect import done_keys

    path = tmp_path / "rows.jsonl"
    path.write_text(
        json.dumps(
            {
                "origin_split": "train",
                "task_id": 1,
                "trial": 0,
                "tool_name": "x",
                "verbalized": 1.0,
            }
        )
        + "\n"
        + json.dumps(
            {
                "kind": "traj_done",
                "origin_split": "train",
                "task_id": 2,
                "trial": 0,
                "reward": 1.0,
                "n_records": 0,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    done = done_keys(str(path))
    assert ("train", 1, 0) in done
    assert ("train", 2, 0) in done  # marker counts even with zero rows


def test_uninstall_restores_real_litellm(monkeypatch, tmp_path, make_response):
    from gate import config, llm

    monkeypatch.setattr(config, "CACHE_DIR", str(tmp_path))
    real_before = litellm.completion
    llm.tracker = llm.CostTracker()

    def fake(**kwargs):
        return make_response(content="x")

    install(backend=fake)
    assert litellm.completion is not real_before
    uninstall()
    assert litellm.completion is real_before  # not the fake
