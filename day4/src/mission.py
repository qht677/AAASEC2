"""Protected MCP data + agent-written shell computation + LangSmith trace."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

from deepagents import create_deep_agent
from dotenv import load_dotenv
from fastmcp import Client
from fastmcp.client.auth import BearerAuth

from shell_agent import SYSTEM_PROMPT, llm, make_backend

DAY4_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(DAY4_ROOT / ".env")

MCP_URL = os.getenv("MCP_URL", "http://localhost:8002/mcp")
ADMIN_TOKEN = os.getenv("MCP_ADMIN_TOKEN", "admin-secret-token")


def fetch_internal_report() -> str:
    """Fetch the protected quarterly report from the secure MCP server."""

    async def _call() -> str:
        async with Client(MCP_URL, auth=BearerAuth(token=ADMIN_TOKEN)) as client:
            result = await client.call_tool("get_internal_report", {})
            return json.dumps(result.data)

    return asyncio.run(_call())


MISSION = (
    "1. Call fetch_internal_report exactly once to get the protected quarterly data. "
    "2. Write analyze.py inside the assigned workspace. The program must compute total "
    "revenue, total costs, total profit, and each month's profit margin percentage. "
    "3. Execute analyze.py with Python. 4. Report exactly what the program printed, "
    "followed by one evidence-based insight; do not recompute the figures mentally."
)


def main() -> None:
    backend, cleanup = make_backend()
    try:
        agent = create_deep_agent(
            model=llm,
            system_prompt=SYSTEM_PROMPT,
            tools=[fetch_internal_report],
            backend=backend,
        )
        result = agent.invoke(
            {"messages": [{"role": "user", "content": MISSION}]}
        )
        print(result["messages"][-1].content)
    finally:
        cleanup()


if __name__ == "__main__":
    main()
