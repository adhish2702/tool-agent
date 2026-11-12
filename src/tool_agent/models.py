"""Provider-neutral types shared by the agent loop and the LLM providers."""

from dataclasses import dataclass, field

SYSTEM_PROMPT = (
    "You answer questions using two tools. Use `calculate` for every arithmetic step; never do math "
    "yourself. Use `search_notes` to look things up in the user's notes. When you use a note, name the "
    "file it came from. If the notes do not contain the answer, say so instead of guessing."
)


class AgentError(Exception):
    """The agent could not finish (turn limit reached, or the model returned nothing usable)."""


@dataclass
class ToolCall:
    id: str
    name: str
    input: dict


@dataclass
class ToolResult:
    call: ToolCall
    output: str
    is_error: bool


@dataclass
class Turn:
    """One model response: any text, plus the tool calls it wants run."""

    text: str
    tool_calls: list = field(default_factory=list)
