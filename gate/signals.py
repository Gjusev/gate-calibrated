"""Confidence signals for a proposed write action (plan section 4.4).

1. verbalized: one extra call to the same model with the conversation and
   the proposed action, asking for a probability in [0, 1]. Temperature 0.
2. agreement: k sampled next-actions from the same conversation; value is
   the fraction whose canonical action matches the proposal.

Token usage of every signal call is returned so the caller can attribute
cost. No logprobs (Claude does not return them; GLM through this endpoint
was not assumed to either).
"""

import re
from typing import Any, Dict, List, Optional, Tuple

from litellm import completion
from tau_bench.agents.tool_calling_agent import message_to_action

from gate import config
from gate.labeling import action_key


def _usage(response) -> Dict[str, int]:
    u = getattr(response, "usage", None)
    return {
        "prompt": getattr(u, "prompt_tokens", 0) or 0,
        "completion": getattr(u, "completion_tokens", 0) or 0,
    }


_ASSESSOR_PROMPT = (
    "The assistant above is a customer service agent about to execute this "
    "tool call:\n{action_json}\n\n"
    "Estimate the probability that this exact call is correct according to "
    "the store policy and the user's request. Answer with a single number "
    "between 0 and 1 and nothing else."
)

# Structured output (plan 4.4): the model reports the probability through a
# tool call instead of free text. Free-text replies are a fallback.
_PROBABILITY_TOOL = [
    {
        "type": "function",
        "function": {
            "name": "report_probability",
            "description": "Report the estimated probability",
            "parameters": {
                "type": "object",
                "properties": {
                    "probability": {
                        "type": "number",
                        "description": "probability the action is correct, between 0 and 1",
                    }
                },
                "required": ["probability"],
            },
        },
    }
]


def verbalized_confidence(
    messages: List[Dict[str, Any]],
    tool_name: str,
    kwargs: Dict[str, Any],
    model: Optional[str] = None,
) -> Tuple[float, Dict[str, int]]:
    """Probability the model assigns to its own proposed action."""
    import json as _json

    prompt = _ASSESSOR_PROMPT.format(
        action_json=_json.dumps({"name": tool_name, "arguments": kwargs})
    )
    res = completion(
        model=model or config.AGENT_MODEL,
        custom_llm_provider=config.PROVIDER,
        temperature=0,
        max_tokens=1024,
        messages=messages + [{"role": "user", "content": prompt}],
        tools=_PROBABILITY_TOOL,
    )
    msg = res.choices[0].message
    value = None
    if msg.tool_calls:
        try:
            args = msg.tool_calls[0].function.arguments
            if isinstance(args, str):
                args = _json.loads(args or "{}")
            value = float(args.get("probability"))
        except Exception:
            value = None
    if value is None:
        value = parse_probability(msg.content or "")
    return min(max(value, 0.0), 1.0), _usage(res)


def parse_probability(text: str) -> float:
    """Fallback parser for free-text replies: the LAST number wins (models
    that echo the scale, like "on a scale from 0 to 1: 0.8", parse as 0.8
    instead of 0). 0.5 if nothing parses. '95%' -> 0.95.
    """
    matches = re.findall(r"(\d+(?:\.\d+)?)\s*(%)?", text.replace(",", "."))
    if not matches:
        return 0.5
    value = float(matches[-1][0])
    if matches[-1][1] == "%" or value > 1:
        value = value / 100.0 if value <= 100 else 0.5
    return min(max(value, 0.0), 1.0)


def agreement(
    messages: List[Dict[str, Any]],
    tools_info: List[Dict[str, Any]],
    tool_name: str,
    kwargs: Dict[str, Any],
    k: int = None,
    temperature: float = None,
    model: Optional[str] = None,
) -> Tuple[float, List[Dict[str, int]]]:
    """Fraction of k sampled next-actions identical to the proposal."""
    k = config.AGREEMENT_K if k is None else k
    temperature = (
        config.AGREEMENT_TEMPERATURE if temperature is None else temperature
    )
    key = action_key(tool_name, kwargs)
    matches = 0
    usages: List[Dict[str, int]] = []
    for _ in range(k):
        res = completion(
            model=model or config.AGENT_MODEL,
            custom_llm_provider=config.PROVIDER,
            messages=messages,
            tools=tools_info,
            temperature=temperature,
        )
        usages.append(_usage(res))
        try:
            action = message_to_action(res.choices[0].message.model_dump())
            if action_key(action.name, action.kwargs) == key:
                matches += 1
        except Exception:
            pass  # unparsable sample counts as disagreement
    return matches / k, usages
