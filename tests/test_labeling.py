import pytest

from gate.labeling import (
    action_key,
    canonical,
    is_correct,
    write_actions_from_messages,
)


def test_gold_list_replays_all_correct():
    """Mandatory test (plan 4.3): a task's gold list labels all as correct."""
    from tau_bench.envs.retail.tasks_test import TASKS_TEST
    from tau_bench.envs.retail.tasks_train import TASKS_TRAIN

    for tasks in (TASKS_TRAIN[:50], TASKS_TEST[:50]):
        for task in tasks:
            for a in task.actions:
                assert is_correct(a.name, a.kwargs, task.actions), (
                    task.instruction[:60],
                    a.name,
                )


def test_exact_match_and_miss():
    gold = [{"name": "modify_user_address", "kwargs": {"address1": "1 A St"}}]
    assert is_correct("modify_user_address", {"address1": "1 A St"}, gold)
    assert not is_correct("modify_user_address", {"address1": "2 B St"}, gold)
    assert not is_correct("cancel_pending_order", {"address1": "1 A St"}, gold)


def test_item_lists_order_insensitive():
    a = {"order_id": "#1", "item_ids": ["2", "1"], "payment_method_id": "cc"}
    b = {"order_id": "#1", "item_ids": ["1", "2"], "payment_method_id": "cc"}
    assert canonical(a) == canonical(b)


def test_paired_lists_are_state_equivalent_under_reordering():
    # modify_pending_order_items / exchange_delivered_order_items pair
    # item_ids[i] -> new_item_ids[i], but the resulting DB state does not
    # depend on the pairing: the price difference is a sum over pairs, items
    # are replaced set-wise, and the exchange tool even stores
    # sorted(item_ids) in the DB. So both lists are canonicalized
    # independently, and a consistent reorder of both is the same action.
    a = {"item_ids": ["1", "2"], "new_item_ids": ["x", "y"]}
    b = {"item_ids": ["2", "1"], "new_item_ids": ["y", "x"]}
    assert canonical(a) == canonical(b)
    # a different pairing with the same elements IS a different action:
    # e.g. swap 1->y, 2->x instead of 1->x, 2->y exchanges different
    # variants (only accidentally state-equal when prices coincide), and
    # validation can reject one pairing but accept the other.
    c = {"item_ids": ["1", "2"], "new_item_ids": ["y", "x"]}
    d = {"item_ids": ["1", "2"], "new_item_ids": ["x", "y"]}
    assert canonical(c) != canonical(d)


def test_strings_stripped_and_integrals_cast():
    assert canonical({"a": " x "}) == canonical({"a": "x"})
    assert canonical({"n": 5.0}) == canonical({"n": 5})


def test_write_actions_from_messages_counts_and_filters():
    msgs = [
        {"role": "system", "content": "wiki"},
        {"role": "user", "content": "hi"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "function": {
                        "name": "get_order_details",
                        "arguments": '{"order_id": "#1"}',
                    }
                }
            ],
        },
        {"role": "tool", "content": "..."},
        {"role": "assistant", "content": "one moment"},  # respond step
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "function": {
                        "name": "cancel_pending_order",
                        "arguments": '{"order_id": "#1"}',
                    }
                }
            ],
        },
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "function": {
                        "name": "think",
                        "arguments": "{}",
                    }
                },
                {  # second call in same message: never executed by tau-bench
                    "function": {
                        "name": "modify_user_address",
                        "arguments": '{"address1": "x"}',
                    }
                },
            ],
        },
    ]
    out = write_actions_from_messages(msgs)
    # only the irreversible cancel is gated; think passes, and the second
    # tool call in its message is not executed by tau-bench so it is dropped
    assert out == [("cancel_pending_order", {"order_id": "#1"}, 2)]
