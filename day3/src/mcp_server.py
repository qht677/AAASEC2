"""FastMCP server exposing callable tools and downloadable skills."""

from __future__ import annotations

import os
import re
from pathlib import Path

from dotenv import load_dotenv
from fastmcp import FastMCP
from fastmcp.server.providers.skills import SkillsDirectoryProvider

from src.agent import calculate as safe_calculate

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

mcp = FastMCP(f"{os.getenv('STUDENT_NAME', 'qht677')} Tools")


@mcp.tool
def calculate(expression: str) -> float:
    """Safely evaluate basic arithmetic, for example ``2 * (3 + 4) ** 2``."""

    return safe_calculate(expression)


@mcp.tool
def word_stats(text: str) -> dict[str, int]:
    """Count words, unique words, characters, and lines in supplied text."""

    words = re.findall(r"[\w'-]+", text, flags=re.UNICODE)
    return {
        "words": len(words),
        "unique_words": len({word.casefold() for word in words}),
        "characters": len(text),
        "lines": len(text.splitlines()) if text else 0,
    }


mcp.add_provider(SkillsDirectoryProvider(roots=PROJECT_ROOT / "skills"))


if __name__ == "__main__":
    mcp.run(transport="http", host="0.0.0.0", port=8001)
