"""The agent loop. It is provider-neutral: any object with start / next_turn / add_tool_results works."""

from dataclasses import dataclass, field
from pathlib import Path

from .models import AgentError, ToolResult
from .tools import ToolError, run_tool

__all__ = ["AgentError", "AgentResult", "run_agent"]


@dataclass
class AgentResult:
    answer: str
    steps: list = field(default_factory=list)  # one dict per tool call: name, input, output, is_error


def run_agent(provider, question: str, notes_dir: Path, max_turns: int = 6) -> AgentResult:
    provider.start(question)
    steps = []

    for _ in range(max_turns):
        turn = provider.next_turn()
        if not turn.tool_calls:
            return AgentResult(answer=turn.text, steps=steps)

        results = []
        for call in turn.tool_calls:
            try:
                output, is_error = run_tool(call.name, call.input, notes_dir), False
            except ToolError as exc:
                output, is_error = f"error: {exc}", True
            steps.append({"name": call.name, "input": call.input, "output": output, "is_error": is_error})
            results.append(ToolResult(call=call, output=output, is_error=is_error))
        provider.add_tool_results(results)

    raise AgentError(f"no final answer after {max_turns} turns")
