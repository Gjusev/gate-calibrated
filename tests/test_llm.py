import pytest

from gate import config, llm


@pytest.fixture()
def cache_env(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setattr(config, "MAX_USD", 10.0)
    monkeypatch.setattr(
        config, "PRICES", {"fake": {"input": 1.0, "output": 2.0, "cache_read": 0.1}}
    )
    llm.tracker = llm.CostTracker()
    yield
    llm.uninstall()


def test_temperature0_calls_are_cached(cache_env, make_response):
    calls = []

    def fake(**kwargs):
        calls.append(kwargs)
        return make_response(content="hi")

    llm.install(backend=fake)
    from litellm import completion

    a = completion(model="fake", temperature=0, messages=[{"role": "user", "content": "x"}])
    b = completion(model="fake", temperature=0, messages=[{"role": "user", "content": "x"}])
    assert len(calls) == 1  # second was a cache hit
    assert a.choices[0].message.content == "hi"
    assert b.choices[0].message.content == "hi"
    assert llm.tracker.cache_hits == 1


def test_sampled_calls_not_cached(cache_env, make_response):
    calls = []

    def fake(**kwargs):
        calls.append(kwargs)
        return make_response(content="hi")

    llm.install(backend=fake)
    from litellm import completion

    completion(model="fake", temperature=0.7, messages=[{"role": "user", "content": "x"}])
    completion(model="fake", temperature=0.7, messages=[{"role": "user", "content": "x"}])
    assert len(calls) == 2


def test_budget_guard(cache_env, monkeypatch, make_response):
    monkeypatch.setattr(config, "MAX_USD", 0.0001)
    llm.tracker = llm.CostTracker()

    def fake(**kwargs):
        return make_response(usage=(1000, 0))

    llm.install(backend=fake)
    from litellm import completion

    with pytest.raises(llm.BudgetExceeded):
        completion(model="fake", temperature=0.7, messages=[{"role": "user", "content": "x"}])


def test_insufficient_balance_fails_fast(cache_env):
    import litellm

    calls = []

    def fake(**kwargs):
        calls.append(1)
        raise litellm.RateLimitError(
            "Insufficient balance or no resource package. Please recharge.",
            model="fake",
            llm_provider="openai",
        )

    llm.install(backend=fake)
    from litellm import completion

    with pytest.raises(litellm.RateLimitError):
        completion(model="fake", temperature=0.7, messages=[{"role": "user", "content": "x"}])
    assert len(calls) == 1  # no ladder spin on a non-transient error


def test_usage_accounted_per_model(cache_env, make_response):
    def fake(**kwargs):
        return make_response(usage=(1000, 100))

    llm.install(backend=fake)
    from litellm import completion

    completion(model="fake", temperature=0.7, messages=[{"role": "user", "content": "x"}])
    snap = llm.tracker.snapshot()
    assert snap["fake"]["prompt"] == 1000
    assert snap["fake"]["completion"] == 100
    assert abs(snap["fake"]["usd"] - (1000 * 1.0 / 1e6 + 100 * 2.0 / 1e6)) < 1e-9
