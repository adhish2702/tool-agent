# tool-agent

A small LLM agent that answers questions by calling tools: a safe calculator and a keyword search over your own markdown notes. It works with **Google Gemini (free tier)** or **Claude**. The focus is on **verifiable behavior**: the agent loop and both providers are covered by deterministic tests that run without an API key.

## How it works

```
question -> provider (Gemini or Claude) -> tool call? -> run tool -> result -> provider -> ... -> final answer
```

- `agent.py`: the loop. It is provider-neutral and stops at a turn limit, raising `AgentError` instead of looping forever.
- `providers.py`: adapters that translate between the loop and each API's message format (Gemini `function_call` / `function_response` parts, Claude `tool_use` / `tool_result` blocks).
- `tools.py`: the tools. `calculate` parses arithmetic with `ast` and never calls `eval()`. `search_notes` greps `*.md` files and returns `file:line` matches.
- Tool failures (bad input, division by zero, unknown tool) go back to the model as error results, so it can recover instead of crashing.

## Setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    Mac/Linux: source .venv/bin/activate
pip install -e ".[gemini,dev]"
```

Use `".[anthropic,dev]"` instead (or in addition) to use Claude.

Get a free Gemini key at [Google AI Studio](https://aistudio.google.com/apikey), then set it:

```powershell
# Windows PowerShell
$env:GEMINI_API_KEY="your-key"
```
```bash
# Mac/Linux
export GEMINI_API_KEY=your-key
```

For Claude, set `ANTHROPIC_API_KEY` and pass `--provider anthropic`.

## Run

```bash
python -m tool_agent.cli "What do my notes say about K-Means, and what is 15% of 2340?" --verbose
```

Put your own notes in `notes/` (two sample files are included). Set `TOOL_AGENT_MODEL` to change the model (defaults: `gemini-2.5-flash`, `claude-sonnet-5-5`).

## Test

```bash
pytest
```

The tests use fake clients, so they are fast and need no network. They check that tool results go back in each API's expected format, that tool errors do not crash the loop, that Gemini's automatic function calling is disabled so the loop stays in our control, and that the turn limit works. Gemini tests are skipped if `google-genai` is not installed.

## Design decisions

- **No `eval()`.** Model-generated input is untrusted, so the calculator only allows numbers and arithmetic operators, and caps exponents.
- **Provider abstraction.** The loop depends on three methods (`start`, `next_turn`, `add_tool_results`), so adding another provider means writing one small class.
- **Dependency injection.** Providers take the API client as an argument, which is what makes the tests deterministic.
- **Own the loop.** Gemini's automatic function calling is turned off so every tool call can be logged, validated and capped.
- **Errors are data.** A failed tool call is reported to the model rather than raised.

## Limitations

- Notes search is plain keyword matching, not semantic search.
- There is no conversation memory between runs.
- Tests cover the loop, tools and message formats, not the model's answer quality. Judging that would need a separate evaluation set.
- Free-tier limits and model availability change, so check Google's current rate-limit page if you see 429 errors.

## Ideas to extend

- Add a tool that reads a CSV (for example your FootStats data) and answers stats questions.
- Add a small evaluation set of questions with known answers and track the pass rate.
