"""Integration test of the instrumented loop with a scripted model over the
real retail env (human user strategy: no model calls, offline)."""

from tau_bench.envs.retail.env import MockRetailDomainEnv

from gate.middleware import CollectAgent

# train task 0 gold: return_delivered_order_items #W6067464
# item_ids [9624127908, 8917609800] with credit_card_4190576
GOLD = {
    "order_id": "#W6067464",
    "item_ids": ["9624127908", "8917609800"],
    "payment_method_id": "credit_card_4190576",
}
WRONG = dict(GOLD, item_ids=["1111111111"])


def _make_fake(agent_turns, make_response, verbalized="0.4", n_matching=2):
    state = {"agent_turn": 0, "agreement_samples": 0}

    def _is_assessor(kwargs):
        tools = kwargs.get("tools") or []
        return bool(tools) and tools[0]["function"]["name"] == "report_probability"

    def fake(**kwargs):
        tools = kwargs.get("tools")
        if tools and kwargs.get("temperature") == 0.7:
            # agreement sample: match the proposal n_matching times of five
            state["agreement_samples"] += 1
            proposal = agent_turns[1]  # the gated proposal
            if state["agreement_samples"] <= n_matching:
                return make_response(tool_call=proposal)
            return make_response(content="I will check something else")
        if _is_assessor(kwargs):
            return make_response(content=verbalized)
        name, kw = agent_turns[state["agent_turn"]]
        state["agent_turn"] += 1
        return make_response(tool_call=(name, kw))

    return fake


def _setup(monkeypatch, fake):
    import gate.middleware as mw
    import gate.signals as sig

    monkeypatch.setattr(mw, "completion", fake)
    monkeypatch.setattr(sig, "completion", fake)
    # human user strategy reads the first message with input(); answer it
    monkeypatch.setattr("builtins.input", lambda prompt="": "hello")


def test_collect_agent_records_signals_and_labels(monkeypatch, make_response):
    agent_turns = [
        ("get_order_details", {"order_id": "#W6067464", "user_id": "omar_anderson_3203"}),
        ("return_delivered_order_items", WRONG),
        ("think", {"thought": "wrap up"}),
    ]
    _setup(monkeypatch, _make_fake(agent_turns, make_response))

    env = MockRetailDomainEnv(
        user_strategy="human", task_split="train", task_index=0
    )
    agent = CollectAgent(
        tools_info=env.tools_info, wiki=env.wiki, model="fake", provider="x"
    )
    agent.split_label = "fit"
    agent.trial = 0
    result, records = agent.solve(env, task_index=0, max_num_steps=3)

    assert len(records) == 1
    r = records[0]
    assert r.tool_name == "return_delivered_order_items"
    assert r.tier == "irreversible"
    assert r.correct is False  # WRONG item ids do not match gold
    assert r.verbalized == 0.4
    assert r.agreement == 0.4
    assert r.step_index == 1  # one previous agent turn (the read call)
    assert r.task_id == 0
    assert r.signal_usage["verbalized"]["prompt"] == 10
    assert len(r.signal_usage["agreement"]) == 5
    # the wrong action still executed (base agent behavior)
    assert any(
        a.name == "return_delivered_order_items" for a in env.actions
    )


def test_collect_agent_correct_action_labels_true(monkeypatch, make_response):
    agent_turns = [
        ("get_order_details", {"order_id": "#W6067464", "user_id": "omar_anderson_3203"}),
        ("return_delivered_order_items", GOLD),
        ("think", {"thought": "done"}),
    ]
    _setup(
        monkeypatch,
        _make_fake(agent_turns, make_response, verbalized="0.99", n_matching=5),
    )

    env = MockRetailDomainEnv(
        user_strategy="human", task_split="train", task_index=0
    )
    agent = CollectAgent(
        tools_info=env.tools_info, wiki=env.wiki, model="fake", provider="x"
    )
    result, records = agent.solve(env, task_index=0, max_num_steps=3)

    assert len(records) == 1
    assert records[0].correct is True
    assert records[0].verbalized == 0.99
    assert records[0].agreement == 1.0
