"""Read-only tools; neither arbitrary Python nor shell commands are executed."""

import math
import re
from pathlib import Path

from langchain_core.tools import tool

from agent_native.tools import ToolError, ToolRegistry


def build_tools(knowledge_dir: Path):
    root = knowledge_dir.expanduser().resolve()

    @tool
    def calculator(expression: str) -> dict:
        """Calculate arithmetic with +, -, *, /, //, %, parentheses. No powers or code."""
        try:
            if "**" in expression:
                raise ToolError("거듭제곱은 지원하지 않습니다.")
            result = ToolRegistry._calculate({"expression": expression}, {})
            if not isinstance(result["result"], (int, float)) or not math.isfinite(result["result"]):
                raise ToolError("유한한 실수 결과만 지원합니다.")
            return result
        except (ToolError, ValueError, OverflowError, RecursionError) as exc:
            return {"error": str(exc)}

    @tool
    def search_knowledge(query: str) -> dict:
        """Search local Markdown policies by keywords (e.g. 휴가). Return cited excerpts."""
        tokens = set(re.findall(r"[\w가-힣]+", query.lower()))
        if not tokens:
            return {"matches": []}
        matches = []
        # Explicitly bounded, top-level Markdown corpus. Symlinks are excluded.
        for path in sorted(root.glob("*.md"))[:100]:
            if path.is_symlink() or not path.is_file():
                continue
            try:
                if path.stat().st_size > 1_000_000:
                    continue
                content = path.read_text(encoding="utf-8")
            except (OSError, UnicodeError):
                continue
            for chunk in re.split(r"\n(?=##?\s)", content):
                score = sum(token in chunk.lower() for token in tokens)
                if score:
                    matches.append({"source": path.name, "score": score, "content": chunk[:1200]})
        matches.sort(key=lambda item: item["score"], reverse=True)
        return {"matches": matches[:3]}

    return [calculator, search_knowledge]
