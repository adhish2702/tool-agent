"""Tools the agent can call: a safe calculator and a keyword search over local notes."""

import ast
import operator
from pathlib import Path

MAX_EXPONENT = 1000
MAX_SEARCH_RESULTS = 5


class ToolError(Exception):
    """Raised when a tool is called with bad input. Reported to the model, not fatal."""


_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def _eval(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        left, right = _eval(node.left), _eval(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > MAX_EXPONENT:
            raise ToolError(f"exponent too large (max {MAX_EXPONENT})")
        try:
            return _BIN_OPS[type(node.op)](left, right)
        except ZeroDivisionError:
            raise ToolError("division by zero") from None
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_eval(node.operand))
    raise ToolError("only numbers and + - * / // % ** are allowed")


def calculate(expression: str) -> str:
    """Evaluate an arithmetic expression without using eval()."""
    try:
        tree = ast.parse(expression.strip(), mode="eval")
    except SyntaxError:
        raise ToolError("invalid expression") from None
    return str(_eval(tree.body))


def search_notes(query: str, notes_dir: Path) -> str:
    """Case-insensitive keyword search over *.md files. Returns file:line matches."""
    query = query.strip().lower()
    if not query:
        raise ToolError("query must not be empty")
    hits = []
    for path in sorted(Path(notes_dir).glob("*.md")):
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if query in line.lower():
                hits.append(f"{path.name}:{number}: {line.strip()}")
                if len(hits) == MAX_SEARCH_RESULTS:
                    return "\n".join(hits)
    return "\n".join(hits) if hits else "no matches"


TOOL_SPECS = [
    {
        "name": "calculate",
        "description": "Evaluate an arithmetic expression exactly. Use for any math instead of computing in your head.",
        "input_schema": {
            "type": "object",
            "properties": {"expression": {"type": "string", "description": "e.g. '(12 + 8) * 3 / 4'"}},
            "required": ["expression"],
        },
    },
    {
        "name": "search_notes",
        "description": "Search the user's markdown notes for a keyword. Returns up to 5 'file:line: text' matches.",
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "keyword or phrase to look for"}},
            "required": ["query"],
        },
    },
]


def run_tool(name: str, tool_input: dict, notes_dir: Path) -> str:
    """Dispatch a tool call. Raises ToolError for unknown tools or bad input."""
    try:
        if name == "calculate":
            return calculate(tool_input["expression"])
        if name == "search_notes":
            return search_notes(tool_input["query"], notes_dir)
    except (KeyError, TypeError, AttributeError):
        raise ToolError(f"bad input for {name}") from None
    raise ToolError(f"unknown tool: {name}")
