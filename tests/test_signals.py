import pytest

from gate.signals import agreement, parse_probability, verbalized_confidence


@pytest.mark.parametrize(
    "text,expected",
    [
        ("0.87", 0.87),
        (" 0.87 ", 0.87),
        ("95%", 0.95),
        ("87", 0.87),
        ("Probability: 0.25", 0.25),
        ("I am not sure", 0.5),
        ("1.0", 1.0),
    ],
)
def test_parse_probability(text, expected):
    assert parse_probability(text) == pytest.approx(expected)


def test_verbalized_returns_value_and_usage(monkeypatch, make_response):
    import gate.signals as sig

    seen = {}

    def fake(**kwargs):
        seen.update(kwargs)
        return make_response(content="0.42")

    monkeypatch.setattr(sig, "completion", fake)
    value, usage = verbalized_confidence(
        [{"role": "user", "content": "hi"}],
        "cancel_pending_order",
        {"order_id": "#1"},
        model="fake-model",
    )
    assert value == pytest.approx(0.42)
    assert usage == {"prompt": 10, "completion": 5}
    assert seen["model"] == "fake-model"
    assert seen["temperature"] == 0
    assert "cancel_pending_order" in seen["messages"][-1]["content"]


def test_verbalized_prefers_tool_call_over_text(monkeypatch, make_response):
    import gate.signals as sig

    def fake(**kwargs):
        return make_response(
            tool_call=("report_probability", {"probability": 0.37})
        )

    monkeypatch.setattr(sig, "completion", fake)
    value, _ = verbalized_confidence(
        [{"role": "user", "content": "hi"}], "t", {}
    )
    assert value == pytest.approx(0.37)


def test_verbalized_none_content_falls_back(monkeypatch, make_response):
    import gate.signals as sig

    def fake(**kwargs):
        return make_response(content=None)  # thinking-only response

    monkeypatch.setattr(sig, "completion", fake)
    value, _ = verbalized_confidence(
        [{"role": "user", "content": "hi"}], "t", {}
    )
    assert value == 0.5


def test_agreement_fraction(monkeypatch, make_response):
    import gate.signals as sig

    samples = [
        make_response(tool_call=("cancel_pending_order", {"order_id": "#1"})),
        make_response(tool_call=("cancel_pending_order", {"order_id": "#2"})),
        make_response(tool_call=("cancel_pending_order", {"order_id": "#1"})),
        make_response(content="I will help"),  # respond: not a match
        make_response(tool_call=("cancel_pending_order", {"order_id": "#1"})),
    ]
    it = iter(samples)

    def fake(**kwargs):
        return next(it)

    monkeypatch.setattr(sig, "completion", fake)
    value, usages = agreement(
        [{"role": "user", "content": "hi"}],
        [],
        "cancel_pending_order",
        {"order_id": "#1"},
        k=5,
        temperature=0.7,
        model="fake-model",
    )
    assert value == pytest.approx(3 / 5)
    assert len(usages) == 5


def test_agreement_kwargs_canonical_order_insensitive(monkeypatch, make_response):
    import gate.signals as sig

    samples = [
        make_response(
            tool_call=(
                "return_delivered_order_items",
                {"order_id": "#1", "item_ids": ["2", "1"]},
            )
        )
    ] * 5
    it = iter(samples)

    def fake(**kwargs):
        return next(it)

    monkeypatch.setattr(sig, "completion", fake)
    value, _ = agreement(
        [{"role": "user", "content": "hi"}],
        [],
        "return_delivered_order_items",
        {"order_id": "#1", "item_ids": ["1", "2"]},
        k=5,
    )
    assert value == 1.0
