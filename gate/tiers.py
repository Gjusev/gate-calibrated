"""Risk tier per retail tool (plan section 4.2).

Verified against tau_bench/envs/retail/tools/ at commit 59a200c: the registry
has 16 tools. The plan's lists match, plus `think`, which writes nothing and
is classified as special. `respond` is not a tool but the action the agent
takes when it answers the user in plain text; it passes without a gate.
"""

READ_TOOLS = frozenset(
    {
        "find_user_id_by_email",
        "find_user_id_by_name_zip",
        "get_order_details",
        "get_product_details",
        "get_user_details",
        "list_all_product_types",
        "calculate",
    }
)

WRITE_TOOLS = frozenset(
    {
        "modify_pending_order_address",
        "modify_pending_order_items",
        "modify_pending_order_payment",
        "modify_user_address",
    }
)

IRREVERSIBLE_TOOLS = frozenset(
    {
        "cancel_pending_order",
        "exchange_delivered_order_items",
        "return_delivered_order_items",
    }
)

SPECIAL_TOOLS = frozenset({"respond", "think", "transfer_to_human_agents"})

_TIERS = {
    "read": READ_TOOLS,
    "write": WRITE_TOOLS,
    "irreversible": IRREVERSIBLE_TOOLS,
    "special": SPECIAL_TOOLS,
}

# Tools the gate can ever act on (everything the plan 4.3/4.4 labels).
GATED_TIERS = ("write", "irreversible")


def tier_of(tool_name: str) -> str:
    """Return the tier of a tool. Unknown tools raise, so new tools fail loudly."""
    for tier, names in _TIERS.items():
        if tool_name in names:
            return tier
    raise KeyError(f"unknown tool: {tool_name!r}")


def gated(tool_name: str) -> bool:
    """True if the action is labeled and gated (write or irreversible tier)."""
    return tier_of(tool_name) in GATED_TIERS
