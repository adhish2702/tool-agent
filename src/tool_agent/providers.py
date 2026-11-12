"""LLM providers. Each one keeps its own conversation history in its API's native format."""

from .models import SYSTEM_PROMPT, AgentError, ToolCall, Turn
from .tools import TOOL_SPECS


class AnthropicProvider:
    def __init__(self, client, model: str, max_tokens: int = 1024):
        self.client, self.model, self.max_tokens = client, model, max_tokens
        self.messages = []

    def start(self, question: str) -> None:
        self.messages = [{"role": "user", "content": question}]

    def next_turn(self) -> Turn:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=SYSTEM_PROMPT,
            tools=TOOL_SPECS,
            messages=self.messages,
        )
        self.messages.append({"role": "assistant", "content": response.content})
        text = "".join(b.text for b in response.content if b.type == "text")
        calls = [ToolCall(b.id, b.name, b.input) for b in response.content if b.type == "tool_use"]
        return Turn(text=text, tool_calls=calls)

    def add_tool_results(self, results) -> None:
        blocks = [
            {"type": "tool_result", "tool_use_id": r.call.id, "content": r.output, "is_error": r.is_error}
            for r in results
        ]
        self.messages.append({"role": "user", "content": blocks})


class GeminiProvider:
    def __init__(self, client, model: str):
        from google.genai import types  # imported here so Anthropic-only users do not need the SDK

        self.types, self.client, self.model = types, client, model
        declarations = [
            types.FunctionDeclaration(name=s["name"], description=s["description"], parameters=s["input_schema"])
            for s in TOOL_SPECS
        ]
        self.config = types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            tools=[types.Tool(function_declarations=declarations)],
            # we run our own loop so that every tool call can be logged, validated and capped
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )
        self.contents = []
        self._calls_made = 0

    def start(self, question: str) -> None:
        t = self.types
        self.contents = [t.Content(role="user", parts=[t.Part.from_text(text=question)])]

    def next_turn(self) -> Turn:
        response = self.client.models.generate_content(
            model=self.model, contents=self.contents, config=self.config
        )
        if not response.candidates or response.candidates[0].content is None:
            raise AgentError("Gemini returned no content (the response may have been blocked)")
        content = response.candidates[0].content
        self.contents.append(content)  # keep the model's reply unchanged so Gemini can verify it later

        text, calls = [], []
        for part in content.parts or []:
            if part.function_call:
                self._calls_made += 1
                fc = part.function_call
                calls.append(ToolCall(fc.id or f"call_{self._calls_made}", fc.name, dict(fc.args or {})))
            elif part.text and not part.thought:
                text.append(part.text)
        return Turn(text="".join(text), tool_calls=calls)

    def add_tool_results(self, results) -> None:
        t = self.types
        parts = [
            t.Part.from_function_response(
                name=r.call.name, response={"error": r.output} if r.is_error else {"output": r.output}
            )
            for r in results
        ]
        self.contents.append(t.Content(role="user", parts=parts))
