"""Single choke point for every model call: disk cache for deterministic
calls, token and cost accounting, runaway-spending guard (plan sections 5
and 8).

Installs a wrapper around litellm.completion. Caching rule: only
temperature-0 calls are cached (sampled calls must draw fresh). tau-bench's
agent and the user simulator also go through this wrapper, so their usage is
accounted too.
"""

import hashlib
import json
import os
import threading
import time
from typing import Any, Dict, Optional

import litellm
from litellm.types.utils import ModelResponse

from gate import config


def _json_safe(value: Any) -> Any:
    """Best-effort JSON-serializable form, for cache keys of payloads that
    may contain objects (messages are plain dicts, but stay safe)."""
    try:
        json.dumps(value)
        return value
    except TypeError:
        return str(value)


class BudgetExceeded(RuntimeError):
    pass


class CostTracker:
    """Thread-safe per-model token and USD accounting."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.usage: Dict[str, Dict[str, float]] = {}
        self.cache_hits = 0
        self.cache_misses = 0  # cacheable calls that were not on disk
        self.cache_bypasses = 0  # sampled calls, never cacheable

    def add(self, model: str, prompt: int, completion: int, cache_read: int = 0) -> float:
        price = config.PRICES.get(model)
        usd = 0.0
        if price:
            usd = (
                (prompt - cache_read) * price["input"] / 1e6
                + completion * price["output"] / 1e6
                + cache_read * price["cache_read"] / 1e6
            )
        with self.lock:
            row = self.usage.setdefault(
                model,
                {"calls": 0, "prompt": 0, "completion": 0, "cache_read": 0, "usd": 0.0},
            )
            row["calls"] += 1
            row["prompt"] += prompt
            row["completion"] += completion
            row["cache_read"] += cache_read
            row["usd"] += usd
            total = sum(r["usd"] for r in self.usage.values())
        if total > config.MAX_USD:
            raise BudgetExceeded(
                f"spent ${total:.2f} > guard ${config.MAX_USD:.2f}"
            )
        return usd

    def snapshot(self) -> Dict[str, Any]:
        with self.lock:
            snap = {m: dict(v) for m, v in self.usage.items()}
            snap["_total_usd"] = round(sum(r["usd"] for r in self.usage.values()), 4)
            snap["_cache"] = {
                "hits": self.cache_hits,
                "misses": self.cache_misses,
                "bypasses": self.cache_bypasses,
            }
            return snap


tracker = CostTracker()

_original_completion = None
_true_litellm_completion = None


def _cache_key(payload: Dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def _cached_completion(*args: Any, **kwargs: Any) -> ModelResponse:
    """Wrapper around litellm.completion with caching and accounting."""
    model = kwargs.get("model", args[0] if args else "?")
    cacheable = kwargs.get("temperature", None) == 0
    key = None
    if cacheable:
        payload = {
            "args": [_json_safe(a) for a in args],
            "kwargs": {
                k: v
                for k, v in kwargs.items()
                if k not in ("api_key", "api_base", "base_url", "num_retries")
            },
        }
        key = _cache_key(payload)
        path = os.path.join(config.CACHE_DIR, f"{key}.json")
        blob = None
        for _attempt in range(3):  # Windows: a concurrent writer may be
            # replacing the file while we hold it open; retry briefly
            try:
                if os.path.exists(path):
                    with open(path, encoding="utf-8") as f:
                        blob = json.load(f)
                break
            except (PermissionError, json.JSONDecodeError):
                time.sleep(0.2)
        if blob is not None:
            resp = ModelResponse(**blob["response"])
            resp._hidden_params = {"response_cost": blob.get("usd")}
            with tracker.lock:
                tracker.cache_hits += 1
            return resp
        with tracker.lock:
            tracker.cache_misses += 1
    else:
        with tracker.lock:
            tracker.cache_bypasses += 1

    # z.ai rate-limits aggressively under concurrency. litellm's internal
    # retries back off too fast under sustained pressure, so this wrapper
    # adds its own ladder on top.
    kwargs.setdefault("num_retries", 2)
    delays = (4, 8, 16, 32, 64, 120)
    last_err = None
    for attempt in range(len(delays) + 1):
        try:
            resp = _original_completion(*args, **kwargs)
            break
        except litellm.RateLimitError as e:
            # "Insufficient balance" is not transient: no amount of backoff
            # fixes it, so fail fast instead of spinning through the ladder.
            if "insufficient balance" in str(e).lower():
                raise
            last_err = e
            if attempt < len(delays):
                time.sleep(delays[attempt])
    else:
        raise last_err
    usage = getattr(resp, "usage", None)
    prompt = getattr(usage, "prompt_tokens", 0) or 0
    completion = getattr(usage, "completion_tokens", 0) or 0
    cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0
    usd = tracker.add(model, prompt, completion, cache_read)
    if cacheable:
        os.makedirs(config.CACHE_DIR, exist_ok=True)
        tmp = f"{path}.tmp.{os.getpid()}.{threading.get_ident()}"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(
                {"response": resp.model_dump(), "usd": usd, "key": key}, f
            )
        os.replace(tmp, path)
    return resp


def install(backend=None) -> None:
    """Route litellm.completion through the wrapper (idempotent).

    Also rebinds modules that did `from litellm import completion` before
    this call (tau-bench's user simulator does exactly that). `backend`
    overrides the wrapped function (tests inject a fake here); when a
    backend is given, no rebinding happens, since no module holds it.
    """
    global _original_completion, _true_litellm_completion
    if _original_completion is not None:
        return
    orig = backend or litellm.completion
    _original_completion = orig
    _true_litellm_completion = litellm.completion
    litellm.completion = _cached_completion
    if backend is None:
        import sys

        for mod in list(sys.modules.values()):
            if getattr(mod, "completion", None) is orig:
                try:
                    mod.completion = _cached_completion
                except Exception:
                    pass


def uninstall() -> None:
    """Restore the real litellm.completion (tests)."""
    global _original_completion, _true_litellm_completion
    if _original_completion is None:
        return
    import sys

    for mod in list(sys.modules.values()):
        if getattr(mod, "completion", None) is _cached_completion:
            try:
                mod.completion = _true_litellm_completion
            except Exception:
                pass
    litellm.completion = _true_litellm_completion
    _original_completion = None
    _true_litellm_completion = None
