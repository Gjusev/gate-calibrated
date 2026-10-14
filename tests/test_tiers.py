import pytest

from gate.tiers import (
    IRREVERSIBLE_TOOLS,
    READ_TOOLS,
    SPECIAL_TOOLS,
    WRITE_TOOLS,
    gated,
    tier_of,
)


def _retail_tool_names():
    from tau_bench.envs.retail.tools import ALL_TOOLS

    return {t.get_info()["function"]["name"] for t in ALL_TOOLS}


def test_every_retail_tool_has_a_tier():
    names = _retail_tool_names()
    classified = READ_TOOLS | WRITE_TOOLS | IRREVERSIBLE_TOOLS | SPECIAL_TOOLS
    # every registry tool is classified, and the classification knows no
    # tool that does not exist (respond is an action, not a registry tool)
    assert names <= classified
    assert (classified - SPECIAL_TOOLS) <= names


def test_respond_and_think_are_special():
    assert tier_of("respond") == "special"
    assert tier_of("think") == "special"
    assert tier_of("transfer_to_human_agents") == "special"


def test_write_and_irreversible_sets_match_plan():
    assert WRITE_TOOLS == frozenset(
        {
            "modify_pending_order_address",
            "modify_pending_order_items",
            "modify_pending_order_payment",
            "modify_user_address",
        }
    )
    assert IRREVERSIBLE_TOOLS == frozenset(
        {
            "cancel_pending_order",
            "exchange_delivered_order_items",
            "return_delivered_order_items",
        }
    )


def test_gated_only_write_and_irreversible():
    assert gated("modify_user_address")
    assert gated("cancel_pending_order")
    assert not gated("get_order_details")
    assert not gated("respond")


def test_unknown_tool_raises():
    with pytest.raises(KeyError):
        tier_of("send_email")
