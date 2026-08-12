"""Day 4 challenge: protected UAV telemetry analyzed through shell execution."""

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
MY_TOOL_NAME = "get_drone_sensor_readings"


def fetch_my_data() -> str:
    """Fetch protected UAV telemetry from the authenticated MCP server."""

    async def _call() -> str:
        async with Client(MCP_URL, auth=BearerAuth(token=ADMIN_TOKEN)) as client:
            result = await client.call_tool(MY_TOOL_NAME, {})
            return json.dumps(result.data)

    return asyncio.run(_call())


MISSION = (
    "1. Call fetch_my_data exactly once to obtain the protected drone telemetry. "
    "2. Write analyze_sensors.py in the assigned workspace. It must compute average and "
    "maximum motor temperature, minimum battery voltage, and list every timestamp whose "
    "motor temperature exceeds safe_motor_temp_c. "
    "3. Execute the program with Python. 4. Report exactly what the program printed, "
    "plus one operational safety insight; do not invent readings."
)


def main() -> None:
    backend, cleanup = make_backend()
    try:
        agent = create_deep_agent(
            model=llm,
            system_prompt=SYSTEM_PROMPT,
            tools=[fetch_my_data],
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
