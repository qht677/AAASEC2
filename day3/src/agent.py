"""Day 3 agent implementation behind a stable ``build_agent`` boundary."""

from __future__ import annotations

import ast
import asyncio
import math
import operator
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_core.messages import AIMessage

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

_BINARY_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPERATORS = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def calculate(expression: str) -> float:
    """Safely evaluate basic arithmetic, for example ``2 * (3 + 4)``."""

    if not expression.strip() or len(expression) > 200:
        raise ValueError("Expression must contain between 1 and 200 characters.")

    tree = ast.parse(expression, mode="eval")
    if sum(1 for _ in ast.walk(tree)) > 64:
        raise ValueError("Expression is too complex.")

    def evaluate(node: ast.AST) -> int | float:
        if isinstance(node, ast.Expression):
            return evaluate(node.body)
        if isinstance(node, ast.Constant) and type(node.value) in {int, float}:
            return node.value
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPERATORS:
            return _UNARY_OPERATORS[type(node.op)](evaluate(node.operand))
        if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPERATORS:
            left = evaluate(node.left)
            right = evaluate(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > 100:
                raise ValueError("Exponent is too large.")
            return _BINARY_OPERATORS[type(node.op)](left, right)
        raise ValueError(f"Unsupported arithmetic syntax: {type(node).__name__}")

    result = evaluate(tree)
    numeric = float(result)
    if not math.isfinite(numeric) or abs(numeric) > 1e100:
        raise ValueError("Result is outside the supported range.")
    return numeric


def current_time() -> str:
    """Return the current local time with its UTC offset."""

    return datetime.now().astimezone().isoformat(timespec="seconds")


def _message_text(message: Any) -> str:
    if isinstance(message, dict):
        return str(message.get("content", ""))
    return str(getattr(message, "content", message))


class FakeAgent:
    """Deterministic stand-in with the same async invocation interface."""

    async def ainvoke(self, payload: dict[str, Any], *args: Any, **kwargs: Any):
        messages = list(payload.get("messages", []))
        prompt = _message_text(messages[-1]) if messages else ""
        lowered = prompt.lower()

        if "research brief" in lowered:
            reply = (
                "**Headline** — Networked agents need explicit contracts and bounded tools.\n\n"
                "**Context** — Turning an agent into a service makes it reusable by any network "
                "client. Standard HTTP, MCP, and A2A boundaries keep implementations replaceable.\n\n"
                "**Findings**\n"
                "- FastAPI exposes a stable OpenResponses-compatible boundary.\n"
                "- MCP separates callable tools from downloadable procedural skills.\n"
                "- Containers make the service reproducible across machines.\n\n"
                "**Recommendation** — Validate the complete pipeline locally before deployment.\n\n"
                "**Confidence** — High; the response follows the repository's required lab design."
            )
        elif "time" in lowered:
            reply = f"The current local time is {current_time()}."
        else:
            reply = (
                "Fake agent is ready. Ask for a research brief, arithmetic, or the current time. "
                "Set USE_FAKE=0 with an OpenRouter key to use the real Deep Agent."
            )
        return {"messages": [*messages, AIMessage(content=reply)]}


def build_agent() -> Any:
    """Build either the deterministic fake or the real Deep Agent."""

    if os.getenv("USE_FAKE", "0") == "1":
        return FakeAgent()

    from deepagents import create_deep_agent
    from deepagents.backends.filesystem import FilesystemBackend
    from langchain_openai import ChatOpenAI

    model = ChatOpenAI(
        model=os.getenv(
            "OPENROUTER_MODEL", "nvidia/nemotron-3-super-120b-a12b:free"
        ),
        temperature=0,
        base_url="https://openrouter.ai/api/v1",
    )
    system_prompt = (
        "You are qht677's Day 3 networked AI agent. Use calculate for every arithmetic "
        "request and current_time for every time request instead of answering from memory. "
        "Discover and follow relevant skills under /skills/. You have filesystem tools but no "
        "shell execution. Never read or reveal .env files, credentials, tokens, or secrets."
    )
    return create_deep_agent(
        model=model,
        tools=[calculate, current_time],
        system_prompt=system_prompt,
        backend=FilesystemBackend(root_dir=PROJECT_ROOT, virtual_mode=True),
        skills=["/skills/"],
    )


async def _smoke_test() -> None:
    agent = build_agent()
    result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": "What is 17 * 23 and what time is it?"}]}
    )
    print(_message_text(result["messages"][-1]))


if __name__ == "__main__":
    asyncio.run(_smoke_test())
