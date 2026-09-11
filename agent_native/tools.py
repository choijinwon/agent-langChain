from __future__ import annotations

import ast
import operator
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from .models import new_id, now_iso
from .store import JsonStateStore


class ToolError(Exception):
    pass


@dataclass(frozen=True, slots=True)
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]]
    requires_approval: bool = False
    risk: str = "read"

    def spec(self) -> dict[str, Any]:
        return {
            "type": "function",
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
            "strict": True,
        }


class ToolRegistry:
    def __init__(self, store: JsonStateStore, knowledge_dir: Path):
        self.store = store
        self.knowledge_dir = knowledge_dir
        self._tools: dict[str, Tool] = {}
        self._register_defaults()

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def specs(self) -> list[dict[str, Any]]:
        return [tool.spec() for tool in self._tools.values()]

    def execute(self, name: str, arguments: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        tool = self.get(name)
        if not tool:
            raise ToolError(f"Unknown tool: {name}")
        required = tool.parameters.get("required", [])
        missing = [key for key in required if key not in arguments]
        if missing:
            raise ToolError(f"Missing required arguments: {', '.join(missing)}")
        return tool.handler(arguments, context)

    def _register_defaults(self) -> None:
        self.register(Tool(
            name="calculator",
            description="Evaluate a basic arithmetic expression.",
            parameters={
                "type": "object",
                "properties": {"expression": {"type": "string"}},
                "required": ["expression"],
                "additionalProperties": False,
            },
            handler=self._calculate,
        ))
        self.register(Tool(
            name="search_knowledge",
            description="Search internal markdown knowledge for policies and company information.",
            parameters={
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
                "additionalProperties": False,
            },
            handler=self._search_knowledge,
        ))
        self.register(Tool(
            name="remember_user_fact",
            description="Save a user-provided preference or fact for future conversations.",
            parameters={
                "type": "object",
                "properties": {
                    "key": {"type": "string"},
                    "value": {"type": "string"},
                },
                "required": ["key", "value"],
                "additionalProperties": False,
            },
            handler=self._remember,
            risk="local_write",
        ))
        self.register(Tool(
            name="create_task",
            description="Create a task in the shared team task list. Requires explicit user approval.",
            parameters={
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "due": {"type": ["string", "null"]},
                },
                "required": ["title", "due"],
                "additionalProperties": False,
            },
            handler=self._create_task,
            requires_approval=True,
            risk="external_write",
        ))

    @staticmethod
    def _calculate(arguments: dict[str, Any], _: dict[str, Any]) -> dict[str, Any]:
        expression = str(arguments["expression"])
        if len(expression) > 100:
            raise ToolError("Expression is too long")
        operators: dict[type[ast.AST], Callable[..., Any]] = {
            ast.Add: operator.add,
            ast.Sub: operator.sub,
            ast.Mult: operator.mul,
            ast.Div: operator.truediv,
            ast.FloorDiv: operator.floordiv,
            ast.Mod: operator.mod,
            ast.Pow: operator.pow,
            ast.USub: operator.neg,
            ast.UAdd: operator.pos,
        }

        def evaluate(node: ast.AST) -> float | int:
            if isinstance(node, ast.Expression):
                return evaluate(node.body)
            if isinstance(node, ast.Constant) and type(node.value) in {int, float}:
                return node.value
            if isinstance(node, ast.BinOp) and type(node.op) in operators:
                left, right = evaluate(node.left), evaluate(node.right)
                if isinstance(node.op, ast.Pow) and abs(right) > 10:
                    raise ToolError("Exponent is too large")
                return operators[type(node.op)](left, right)
            if isinstance(node, ast.UnaryOp) and type(node.op) in operators:
                return operators[type(node.op)](evaluate(node.operand))
            raise ToolError("Only basic arithmetic is supported")

        try:
            result = evaluate(ast.parse(expression, mode="eval"))
        except (SyntaxError, ZeroDivisionError, OverflowError) as exc:
            raise ToolError(str(exc)) from exc
        return {"expression": expression, "result": result}

    def _search_knowledge(self, arguments: dict[str, Any], _: dict[str, Any]) -> dict[str, Any]:
        query = str(arguments["query"]).strip().lower()
        query_tokens = set(re.findall(r"[\w가-힣]+", query))
        matches: list[dict[str, Any]] = []
        if not self.knowledge_dir.exists():
            return {"query": query, "matches": []}
        for path in sorted(self.knowledge_dir.glob("*.md")):
            text = path.read_text(encoding="utf-8")
            chunks = [part.strip() for part in re.split(r"\n(?=##?\s)", text) if part.strip()]
            for chunk in chunks:
                lowered = chunk.lower()
                score = sum(1 for token in query_tokens if token in lowered)
                if score:
                    matches.append({"source": path.name, "score": score, "content": chunk[:700]})
        matches.sort(key=lambda match: match["score"], reverse=True)
        return {"query": query, "matches": matches[:3]}

    def _remember(self, arguments: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        key = re.sub(r"[^\w가-힣_-]", "_", str(arguments["key"]))[:50]
        value = str(arguments["value"])[:500]
        self.store.remember(context["user_id"], key, value)
        return {"saved": True, "key": key, "value": value}

    def _create_task(self, arguments: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        task = {
            "id": new_id("task"),
            "title": str(arguments["title"])[:200],
            "due": arguments.get("due"),
            "created_by": context["user_id"],
            "created_at": now_iso(),
            "status": "open",
        }
        self.store.add_task(task)
        return {"created": True, "task": task}

