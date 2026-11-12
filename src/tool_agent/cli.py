"""Command-line entry point: python -m tool_agent.cli "your question"."""

import argparse
import os
import sys
from pathlib import Path

from .agent import AgentError, run_agent

DEFAULT_MODELS = {"gemini": "gemini-2.5-flash", "anthropic": "claude-sonnet-5-5"}


def build_provider(name: str, model: str):
    if name == "gemini":
        from google import genai  # reads GEMINI_API_KEY or GOOGLE_API_KEY from the environment

        from .providers import GeminiProvider

        return GeminiProvider(genai.Client(), model)

    import anthropic  # reads ANTHROPIC_API_KEY from the environment

    from .providers import AnthropicProvider

    return AnthropicProvider(anthropic.Anthropic(), model)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Ask a question; the agent may use a calculator and your notes.")
    parser.add_argument("question")
    parser.add_argument("--provider", choices=sorted(DEFAULT_MODELS), default="gemini")
    parser.add_argument("--notes", default="notes", help="folder of .md notes (default: ./notes)")
    parser.add_argument("--max-turns", type=int, default=6)
    parser.add_argument("--verbose", action="store_true", help="print each tool call")
    args = parser.parse_args(argv)

    model = os.environ.get("TOOL_AGENT_MODEL", DEFAULT_MODELS[args.provider])

    try:
        provider = build_provider(args.provider, model)
        result = run_agent(provider, args.question, Path(args.notes), args.max_turns)
    except AgentError as exc:
        print(f"agent failed: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # API errors (bad key, quota, network): show one clear line, not a traceback
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    if args.verbose:
        for step in result.steps:
            flag = " (error)" if step["is_error"] else ""
            print(f"[tool] {step['name']}({step['input']}) -> {step['output']}{flag}")
    print(result.answer)
    return 0


if __name__ == "__main__":
    sys.exit(main())
