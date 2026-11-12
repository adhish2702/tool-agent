"""Agent-loop tests. A scripted fake provider stands in for the LLM, so no network or key is needed."""

import pytest

from tool_agent.agent import AgentError, run_agent
from tool_agent.models import ToolCall, Turn


class FakeProvider:
    def __init__(self, turns):
        self.turns = list(turns)
        self.question = None
        self.results_log = []  # what the loop sent back after each turn

    def start(self, question):
        self.question = question

    def next_turn(self):
        return self.turns.pop(0)

    def add_tool_results(self, results):
        self.results_log.append(results)


def call(id_, name, **tool_input):
    return ToolCall(id_, name, tool_input)


def test_answers_directly_without_tools(tmp_path):
    provider = FakeProvider([Turn(text="Hello!")])
    result = run_agent(provider, "hi", tmp_path)
    assert provider.question == "hi"
    assert result.answer == "Hello!"
    assert result.steps == []


def test_runs_tool_and_feeds_result_back(tmp_path):
    provider = FakeProvider(
        [
            Turn(text="", tool_calls=[call("t1", "calculate", expression="6 * 7")]),
            Turn(text="The answer is 42."),
        ]
    )
    result = run_agent(provider, "what is 6 * 7?", tmp_path)

    assert result.answer == "The answer is 42."
    assert result.steps == [
        {"name": "calculate", "input": {"expression": "6 * 7"}, "output": "42", "is_error": False}
    ]
    sent = provider.results_log[0][0]
    assert (sent.call.id, sent.output, sent.is_error) == ("t1", "42", False)


def test_tool_errors_are_reported_to_the_model_not_raised(tmp_path):
    provider = FakeProvider(
        [
            Turn(text="", tool_calls=[call("t1", "calculate", expression="1 / 0")]),
            Turn(text="That is undefined."),
        ]
    )
    result = run_agent(provider, "1/0?", tmp_path)

    assert result.steps[0]["is_error"] is True
    sent = provider.results_log[0][0]
    assert sent.is_error is True
    assert "division by zero" in sent.output


def test_unknown_tool_is_an_error_result(tmp_path):
    provider = FakeProvider([Turn(text="", tool_calls=[call("t1", "rm_rf")]), Turn(text="I cannot do that.")])
    result = run_agent(provider, "do it", tmp_path)
    assert result.steps[0]["is_error"] is True
    assert "unknown tool" in result.steps[0]["output"]


def test_notes_search_uses_given_folder(tmp_path):
    (tmp_path / "n.md").write_text("K-Means needs k chosen in advance\n")
    provider = FakeProvider(
        [Turn(text="", tool_calls=[call("t1", "search_notes", query="k-means")]), Turn(text="See n.md.")]
    )
    result = run_agent(provider, "what do my notes say about k-means?", tmp_path)
    assert result.steps[0]["output"].startswith("n.md:1:")


def test_multiple_tool_calls_in_one_turn_are_all_run(tmp_path):
    provider = FakeProvider(
        [
            Turn(text="", tool_calls=[call("a", "calculate", expression="1 + 1"), call("b", "calculate", expression="2 * 3")]),
            Turn(text="2 and 6."),
        ]
    )
    result = run_agent(provider, "two sums", tmp_path)
    assert [s["output"] for s in result.steps] == ["2", "6"]
    assert [r.call.id for r in provider.results_log[0]] == ["a", "b"]


def test_stops_after_max_turns(tmp_path):
    looping = [Turn(text="", tool_calls=[call(f"t{i}", "calculate", expression="1 + 1")]) for i in range(3)]
    with pytest.raises(AgentError, match="3 turns"):
        run_agent(FakeProvider(looping), "loop forever", tmp_path, max_turns=3)
