"""FastAPI service exposing the agent through an OpenResponses subset."""

from __future__ import annotations

import os
import time
import uuid
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.agent import build_agent

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")


class ResponseRequest(BaseModel):
    input: str = Field(min_length=1, max_length=20_000)
    model: str | None = None


def _content_text(message: Any) -> str:
    content = message.get("content", "") if isinstance(message, dict) else getattr(message, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") in {"text", "output_text"}:
                parts.append(str(block.get("text", "")))
        return "\n".join(part for part in parts if part)
    return str(content)


def _skill_cards() -> list[dict[str, str]]:
    cards = []
    for skill_file in sorted((PROJECT_ROOT / "skills").glob("*/SKILL.md")):
        name = skill_file.parent.name
        description = f"Procedural skill: {name}"
        for line in skill_file.read_text(encoding="utf-8").splitlines():
            if line.startswith("name:"):
                name = line.split(":", 1)[1].strip()
            elif line.startswith("description:"):
                description = line.split(":", 1)[1].strip()
        cards.append({"id": name, "name": name, "description": description})
    return cards


app = FastAPI(
    title="AAASEC2 Day 3 Agent",
    version="1.0.0",
    description="A networked Deep Agent with HTTP, MCP skills, and A2A discovery.",
)
agent = build_agent()


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/responses")
async def create_response(request: ResponseRequest) -> dict[str, Any]:
    result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": request.input}]}
    )
    text = _content_text(result["messages"][-1])
    model = request.model or os.getenv("OPENROUTER_MODEL", "aaasec2-day3-agent")
    return {
        "id": f"resp_{uuid.uuid4().hex}",
        "object": "response",
        "created_at": int(time.time()),
        "status": "completed",
        "model": model,
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": text}],
            }
        ],
    }


@app.get("/.well-known/agent-card.json")
async def agent_card() -> dict[str, Any]:
    student_name = os.getenv("STUDENT_NAME", "qht677")
    public_url = os.getenv("PUBLIC_URL", "http://localhost:8000").rstrip("/")
    return {
        "name": f"{student_name} Day 3 Agent",
        "description": "Research, arithmetic, time, and reusable lab-report skills.",
        "url": f"{public_url}/v1/responses",
        "version": "1.0.0",
        "protocol": "openresponses-subset",
        "capabilities": {"streaming": False},
        "skills": _skill_cards(),
    }
