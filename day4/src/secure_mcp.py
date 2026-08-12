"""Authenticated FastMCP service with public and scoped internal tools."""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from fastmcp import FastMCP
from fastmcp.server.auth import require_scopes
from fastmcp.server.auth.providers.jwt import StaticTokenVerifier

DAY4_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(DAY4_ROOT / ".env")

STUDENT_TOKEN = os.getenv("MCP_STUDENT_TOKEN", "student-secret-token")
ADMIN_TOKEN = os.getenv("MCP_ADMIN_TOKEN", "admin-secret-token")

if STUDENT_TOKEN == ADMIN_TOKEN:
    raise RuntimeError("MCP_STUDENT_TOKEN and MCP_ADMIN_TOKEN must be different.")

verifier = StaticTokenVerifier(
    tokens={
        STUDENT_TOKEN: {
            "client_id": "student",
            "scopes": ["read:public"],
        },
        ADMIN_TOKEN: {
            "client_id": "admin",
            "scopes": ["read:public", "read:internal"],
        },
    }
)

mcp = FastMCP("qht677 Day 4 Secure Tools", auth=verifier)


@mcp.tool
def get_server_time() -> str:
    """Return current UTC time; available to any authenticated client."""

    return datetime.now(timezone.utc).isoformat()


@mcp.tool(auth=require_scopes("read:internal"))
def get_internal_report() -> dict:
    """Return quarterly financial data; requires the internal-read scope."""

    return {
        "quarter": "Q3-2026",
        "months": ["July", "August", "September"],
        "revenue_sar": [412_000, 385_000, 505_000],
        "costs_sar": [298_000, 310_000, 342_000],
        "classification": "internal",
    }


@mcp.tool(auth=require_scopes("read:internal"))
def get_drone_sensor_readings() -> dict:
    """Return protected drone lab telemetry for the student challenge."""

    return {
        "unit": "KFUPM-UAV-07",
        "safe_motor_temp_c": 75.0,
        "readings": [
            {"timestamp": "10:00", "motor_temp_c": 61.2, "battery_v": 16.4},
            {"timestamp": "10:05", "motor_temp_c": 68.9, "battery_v": 15.8},
            {"timestamp": "10:10", "motor_temp_c": 77.3, "battery_v": 15.1},
            {"timestamp": "10:15", "motor_temp_c": 72.5, "battery_v": 14.7},
        ],
        "classification": "internal-lab",
    }


if __name__ == "__main__":
    mcp.run(transport="http", host="0.0.0.0", port=8002)
