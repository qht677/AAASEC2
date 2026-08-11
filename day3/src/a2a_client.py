"""A2A-style discovery and delegation client for class agents."""

from __future__ import annotations

import asyncio
import sys
from typing import Any

import httpx

TIMEOUT_SECONDS = 30.0


async def discover(peer_base_url: str) -> dict[str, Any]:
    """Fetch and display a peer's Agent Card."""

    url = f"{peer_base_url.rstrip('/')}/.well-known/agent-card.json"
    async with httpx.AsyncClient(timeout=TIMEOUT_SECONDS, follow_redirects=True) as client:
        response = await client.get(url)
        response.raise_for_status()
        card = response.json()

    if not isinstance(card, dict) or not isinstance(card.get("url"), str):
        raise ValueError("Peer returned an invalid Agent Card: missing string 'url'.")
    skills = card.get("skills", [])
    skill_names = [
        item.get("name", item.get("id", "unknown")) if isinstance(item, dict) else str(item)
        for item in skills
    ]
    print(f"Discovered: {card.get('name', 'Unnamed agent')}")
    print(f"Skills: {', '.join(skill_names) if skill_names else 'none advertised'}")
    return card


def _extract_output_text(payload: dict[str, Any]) -> str:
    for item in payload.get("output", []):
        if not isinstance(item, dict) or item.get("type") != "message":
            continue
        for block in item.get("content", []):
            if isinstance(block, dict) and block.get("type") == "output_text":
                return str(block.get("text", ""))
    raise ValueError("Peer response did not contain an output_text item.")


async def delegate(card: dict[str, Any], task: str) -> str:
    """Delegate a task to the endpoint advertised by an Agent Card."""

    endpoint = card.get("url")
    if not isinstance(endpoint, str) or not endpoint.startswith(("http://", "https://")):
        raise ValueError("Agent Card contains an invalid delegation URL.")
    async with httpx.AsyncClient(timeout=TIMEOUT_SECONDS, follow_redirects=True) as client:
        response = await client.post(endpoint, json={"input": task})
        response.raise_for_status()
        return _extract_output_text(response.json())


async def main() -> None:
    if len(sys.argv) < 3:
        raise SystemExit(
            "Usage: uv run python src/a2a_client.py http://<peer> \"task for their agent\""
        )
    card = await discover(sys.argv[1])
    answer = await delegate(card, " ".join(sys.argv[2:]))
    print("\nDelegated response:\n" + answer)


if __name__ == "__main__":
    asyncio.run(main())
