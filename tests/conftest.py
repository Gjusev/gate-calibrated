import json

import pytest
from litellm.types.utils import Choices, Message, ModelResponse, Usage


@pytest.fixture()
def make_response():
    """Factory of real litellm ModelResponse objects for faking completions.

    Using the real class matters: gate.llm caches by calling model_dump()
    and reparsing ModelResponse(**blob), so the fakes must round-trip.
    """

    def _mk(content=None, tool_call=None, usage=(10, 5)):
        tool_calls = None
        if tool_call:
            name, kwargs = tool_call
            tool_calls = [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": json.dumps(kwargs),
                    },
                }
            ]
        msg = Message(role="assistant", content=content, tool_calls=tool_calls)
        return ModelResponse(
            id="resp_1",
            choices=[
                Choices(
                    index=0,
                    finish_reason="tool_calls" if tool_call else "stop",
                    message=msg,
                )
            ],
            usage=Usage(
                prompt_tokens=usage[0],
                completion_tokens=usage[1],
                cache_read_input_tokens=0,
            ),
        )

    return _mk
