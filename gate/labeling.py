"""Correctness label for a proposed action, by comparison with the gold
actions of the task (plan section 4.3).

correct = True iff some gold action has the same name and canonically equal
kwargs. Canonical comparison: sorted keys, item lists sorted when the order
does not matter, strings stripped, integral floats cast to int. Actions that
are valid but different from gold count as wrong; that limitation is in the
plan's Limitations section.
"""

import json
from typing import Any, Dict, List, Sequence, Tuple

# List-valued kwargs whose order does not matter for the resulting DB state.
# Verified against the tool code: item swaps, returns and exchanges are
# applied per item and the price difference is summed, so the order of
# application does not change the final state.
ORDER_INSENSITIVE_LIST_KEYS = frozenset(
    {"item_ids", "items", "new_items"}
)

# Paired kwargs (modify/exchange tools): item_ids[i] is replaced by
# new_item_ids[i]. Tool semantics differ (verified in the vendor code):
# - exchange_delivered_order_items stores sorted(item_ids) and
#   sorted(new_item_ids) plus a summed price difference: pairing never
#   reaches the DB, so both lists sort independently.
# - modify_pending_order_items mutates "[item for ...][0]" per pair: with
#   duplicate ids the same element is matched twice, so the pair ORDER
#   changes the final state. Pairs sort as a multiset only when the old
#   ids are unique; with duplicates the sequence stays positional.
TOOLS_WITH_INDEPENDENT_LISTS = frozenset({"exchange_delivered_order_items"})
PAIRED_LIST_KEYS = ("item_ids", "new_item_ids")


def _normalize(value: Any, key: str = "", tool_name: str = "") -> Any:
    if isinstance(value, dict):
        if (
            tool_name in TOOLS_WITH_INDEPENDENT_LISTS
            and all(isinstance(value.get(k), list) for k in PAIRED_LIST_KEYS)
        ):
            rest = {
                k: _normalize(v, k, tool_name)
                for k, v in value.items()
                if k not in PAIRED_LIST_KEYS
            }
            return {
                "item_ids": sorted(
                    (_normalize(v, "", tool_name) for v in value["item_ids"]),
                    key=canonical,
                ),
                "new_item_ids": sorted(
                    (_normalize(v, "", tool_name) for v in value["new_item_ids"]),
                    key=canonical,
                ),
                **rest,
            }
        if (
            tool_name not in TOOLS_WITH_INDEPENDENT_LISTS
            and all(isinstance(value.get(k), list) for k in PAIRED_LIST_KEYS)
            and len(value["item_ids"]) == len(value["new_item_ids"])
        ):
            old, new = value["item_ids"], value["new_item_ids"]
            rest = {
                k: _normalize(v, k, tool_name)
                for k, v in value.items()
                if k not in PAIRED_LIST_KEYS
            }
            norm_old = [_normalize(v, "", tool_name) for v in old]
            norm_new = [_normalize(v, "", tool_name) for v in new]
            if len(set(map(json.dumps, norm_old))) == len(norm_old):
                pairs = sorted(
                    zip(norm_old, norm_new),
                    key=lambda p: json.dumps(p, sort_keys=True),
                )
            else:  # duplicates: pair order changes the DB state, keep it
                pairs = list(zip(norm_old, norm_new))
            return {"__pairs__": [list(p) for p in pairs], **rest}
        return {k: _normalize(v, k, tool_name) for k, v in value.items()}
    if isinstance(value, list):
        items = [_normalize(v, "", tool_name) for v in value]
        if key in ORDER_INSENSITIVE_LIST_KEYS:
            return sorted(items, key=canonical)
        return items
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def canonical(kwargs: Dict[str, Any], tool_name: str = "") -> str:
    """Canonical string form of action kwargs."""
    return json.dumps(
        _normalize(kwargs, tool_name=tool_name),
        sort_keys=True,
        separators=(",", ":"),
    )


def action_key(name: str, kwargs: Dict[str, Any]) -> Tuple[str, str]:
    return (name, canonical(kwargs, name))


def is_correct(
    name: str, kwargs: Dict[str, Any], gold_actions: Sequence[Any]
) -> bool:
    """True if (name, kwargs) matches some gold action canonically.

    gold_actions: objects with .name and .kwargs (tau_bench Task.actions) or
    plain dicts with those keys.
    """
    key = action_key(name, kwargs)
    for gold in gold_actions:
        g_name = gold["name"] if isinstance(gold, dict) else gold.name
        g_kwargs = gold["kwargs"] if isinstance(gold, dict) else gold.kwargs
        if g_name == name and canonical(g_kwargs, g_name) == key[1]:
            return True
    return False


def write_actions_from_messages(
    messages: List[Dict[str, Any]],
) -> List[Tuple[str, Dict[str, Any], int]]:
    """Extract gated (write/irreversible) actions from an OpenAI-format
    conversation, in order. tau-bench executes only the first tool call of
    each assistant message, so only that one is extracted. Returns tuples
    (name, kwargs, step_index) where step_index counts previous agent steps
    (every assistant message with a tool call or a respond action).
    """
    from gate.tiers import gated  # local import avoids a cycle in tests

    out: List[Tuple[str, Dict[str, Any], int]] = []
    step = 0
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        tool_calls = msg.get("tool_calls") or []
        if tool_calls:
            call = tool_calls[0]
            name = call["function"]["name"]
            args = call["function"].get("arguments", "{}")
            if isinstance(args, dict):
                kwargs = args
            else:
                try:
                    kwargs = json.loads(args or "{}")
                except json.JSONDecodeError:
                    kwargs = {}
            if gated(name):
                out.append((name, kwargs, step))
            step += 1
        elif msg.get("content"):
            step += 1
    return out
