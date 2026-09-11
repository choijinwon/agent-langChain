"""Read-only tools; neither arbitrary Python nor shell commands are executed."""

import math
from pathlib import Path

from langchain_core.tools import tool

from agent_native.tools import ToolError, ToolRegistry
from .knowledge import KnowledgeIndex


def build_tools(knowledge_dir: Path):
    index = KnowledgeIndex(knowledge_dir)

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
        """Search Markdown via LlamaIndex BM25. Use keywords such as 휴가, 보안, 비용; cite sources."""
        return index.search(query)

    return [calculator, search_knowledge]
