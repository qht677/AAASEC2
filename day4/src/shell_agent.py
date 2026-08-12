"""Day 4 Deep Agent with an explicitly selected execution backend."""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

from deepagents import create_deep_agent
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

DAY4_ROOT = Path(__file__).resolve().parents[1]
WORK_DIR = DAY4_ROOT / "work"
load_dotenv(DAY4_ROOT / ".env")

llm = ChatOpenAI(
    model=os.getenv(
        "OPENROUTER_MODEL", "nvidia/nemotron-3-super-120b-a12b:free"
    ),
    temperature=0,
    base_url="https://openrouter.ai/api/v1",
)

SYSTEM_PROMPT = (
    "You are a Python coding assistant with shell access. Work only inside the "
    "provided workspace. Write files with filesystem tools and run them with execute. "
    "When a command fails, read its output, fix the smallest relevant issue, and rerun "
    "until the requested checks pass. Never request, print, or copy credentials, tokens, "
    "SSH material, .env contents, or files outside the assigned workspace."
)


def make_backend() -> tuple[Any, Callable[[], None]]:
    """Return the selected backend and an idempotent cleanup callback."""

    provider = os.getenv("SANDBOX_PROVIDER", "local").strip().lower()

    if provider == "local":
        from deepagents.backends import LocalShellBackend

        WORK_DIR.mkdir(parents=True, exist_ok=True)
        backend = LocalShellBackend(
            root_dir=str(WORK_DIR),
            virtual_mode=True,
            # Deliberately do not inherit API keys or the rest of the host env.
            env={"PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin")},
        )
        return backend, lambda: None

    if provider == "daytona":
        from daytona import Daytona
        from langchain_daytona import DaytonaSandbox

        sandbox = Daytona().create()
        return DaytonaSandbox(sandbox=sandbox), sandbox.stop

    if provider == "langsmith":
        from deepagents.backends import LangSmithSandbox
        from langsmith.sandbox import SandboxClient

        client = SandboxClient()
        sandbox = client.create_sandbox()
        return (
            LangSmithSandbox(sandbox=sandbox),
            lambda: client.delete_sandbox(sandbox.name),
        )

    raise ValueError(
        "Unsupported SANDBOX_PROVIDER. Use 'local', 'daytona', or 'langsmith'."
    )


TASK = (
    "1. Create calculator.py with add, sub, mul, and div functions; div must raise "
    "ZeroDivisionError on zero. "
    "2. Write test_calculator.py with pytest tests for every operation, including zero. "
    "3. Run the tests using 'python -m pytest' through execute; install pytest first if "
    "it is missing. 4. If anything fails, fix it and rerun until green. "
    "5. Report the exact final pytest output."
)


def main() -> None:
    backend, cleanup = make_backend()
    try:
        agent = create_deep_agent(
            model=llm,
            system_prompt=SYSTEM_PROMPT,
            backend=backend,
        )
        result = agent.invoke(
            {"messages": [{"role": "user", "content": TASK}]}
        )
        print(result["messages"][-1].content)
    finally:
        cleanup()


if __name__ == "__main__":
    main()
