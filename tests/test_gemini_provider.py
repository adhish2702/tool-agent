"""Gemini provider tests. They use the real google-genai types, so shape mistakes are caught.
Skipped automatically if google-genai is not installed."""

from types import SimpleNamespace

import pytest

from tool_agent.agent import run_agent
from tool_agent.models import AgentError, ToolCall, ToolResult
from tool_agent.providers import GeminiProvider

genai_types = pytest.importorskip("google.genai.types")


def g_response(*parts):
    content = genai_types.Content(role="model", parts=list(parts))
    return SimpleNamespace(candidates=[SimpleNamespace(content=content)])


def g_call(name, **args):
    return genai_types.Part(function_call=genai_types.FunctionCall(name=name, args=args))


def g_text(s):
    return genai_types.Part(text=s)


class FakeGeminiClient:
    def __init__(self, scripted):
        self.scripted, self.calls = list(scripted), []
        self.models = self

    def generate_content(self, model, contents, config):
        self.calls.append({"model": model, "contents": list(contents), "config": config})
        return self.scripted.pop(0)


def test_gemini_end_to_end_tool_round_trip(tmp_path):
    client = FakeGeminiClient([g_response(g_call("calculate", expression="6 * 7")), g_response(g_text("42."))])
    result = run_agent(GeminiProvider(client, "gem"), "6 * 7?", tmp_path)

    assert result.answer == "42."
    assert result.steps[0]["output"] == "42"
    second_call_contents = client.calls[1]["contents"]
    assert [c.role for c in second_call_contents] == ["user", "model", "user"]
    fr = second_call_contents[-1].parts[0].function_response
    assert fr.name == "calculate" and fr.response == {"output": "42"}


def test_gemini_automatic_function_calling_is_disabled_and_tools_declared():
    provider = GeminiProvider(FakeGeminiClient([]), "gem")
    assert provider.config.automatic_function_calling.disable is True
    names = [d.name for d in provider.config.tools[0].function_declarations]
    assert names == ["calculate", "search_notes"]


def test_gemini_tool_error_goes_back_as_error_response():
    provider = GeminiProvider(FakeGeminiClient([]), "gem")
    provider.start("q")
    call = ToolCall("call_1", "calculate", {"expression": "1 / 0"})
    provider.add_tool_results([ToolResult(call, "error: division by zero", True)])
    fr = provider.contents[-1].parts[0].function_response
    assert fr.response == {"error": "error: division by zero"}


def test_gemini_ids_are_generated_when_the_api_gives_none():
    client = FakeGeminiClient([g_response(g_call("calculate", expression="1 + 1"), g_call("calculate", expression="2 + 2"))])
    provider = GeminiProvider(client, "gem")
    provider.start("q")
    turn = provider.next_turn()
    assert [c.id for c in turn.tool_calls] == ["call_1", "call_2"]
    assert turn.tool_calls[0].input == {"expression": "1 + 1"}


def test_gemini_empty_response_raises_agent_error(tmp_path):
    client = FakeGeminiClient([SimpleNamespace(candidates=[])])
    with pytest.raises(AgentError, match="no content"):
        run_agent(GeminiProvider(client, "gem"), "q", tmp_path)
