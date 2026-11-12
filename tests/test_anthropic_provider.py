"""Anthropic provider tests, using a fake client."""

from types import SimpleNamespace

from tool_agent.agent import run_agent
from tool_agent.providers import AnthropicProvider

# ---------- Anthropic ----------


def a_text(s):
    return SimpleNamespace(type="text", text=s)


def a_tool(id_, name, **tool_input):
    return SimpleNamespace(type="tool_use", id=id_, name=name, input=tool_input)


class FakeAnthropicClient:
    def __init__(self, scripted):
        self.scripted, self.calls = list(scripted), []
        self.messages = self

    def create(self, **kwargs):
        self.calls.append({**kwargs, "messages": list(kwargs["messages"])})  # snapshot
        return SimpleNamespace(content=self.scripted.pop(0))


def test_anthropic_end_to_end_tool_round_trip(tmp_path):
    client = FakeAnthropicClient([[a_tool("t1", "calculate", expression="6 * 7")], [a_text("42.")]])
    result = run_agent(AnthropicProvider(client, "m"), "6 * 7?", tmp_path)

    assert result.answer == "42."
    sent = client.calls[1]["messages"][-1]
    assert sent["role"] == "user"
    assert sent["content"] == [{"type": "tool_result", "tool_use_id": "t1", "content": "42", "is_error": False}]
    assert client.calls[0]["tools"][0]["name"] == "calculate"


def test_anthropic_error_result_is_flagged(tmp_path):
    client = FakeAnthropicClient([[a_tool("t1", "calculate", expression="1 / 0")], [a_text("undefined")]])
    run_agent(AnthropicProvider(client, "m"), "1/0?", tmp_path)
    sent = client.calls[1]["messages"][-1]["content"][0]
    assert sent["is_error"] is True and "division by zero" in sent["content"]
