from __future__ import annotations

import json
import mimetypes
import sys
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from .config import Settings
from .models import new_id
from .providers import DemoProvider, OpenAIResponsesProvider
from .runtime import AgentRuntime
from .store import JsonStateStore
from .tools import ToolRegistry


ROOT = Path(__file__).resolve().parent.parent
WEB_ROOT = ROOT / "web"


def build_runtime(settings: Settings) -> AgentRuntime:
    settings.validate()
    store = JsonStateStore(settings.data_dir / "state.json")
    tools = ToolRegistry(store, settings.knowledge_dir)
    if settings.provider == "ollama":
        from .ollama_provider import OllamaProvider
        from agent_langchain.knowledge import KnowledgeIndex
        from dataclasses import replace
        index = KnowledgeIndex(settings.knowledge_dir)
        search = tools.get("search_knowledge")
        tools.register(replace(search, handler=lambda args, context: index.search(args["query"])))
        provider = OllamaProvider(settings.ollama_model, settings.ollama_base_url)
    elif settings.provider == "openai":
        provider = OpenAIResponsesProvider(
            api_key=settings.api_key or "", model=settings.model, base_url=settings.base_url
        )
    else:
        provider = DemoProvider()
    return AgentRuntime(provider, tools, store, settings.max_steps)


def make_handler(runtime: AgentRuntime, settings: Settings):
    class Handler(BaseHTTPRequestHandler):
        server_version = "NativeAgent/0.1"

        def do_GET(self) -> None:  # noqa: N802
            path = urlparse(self.path).path
            if path == "/api/health":
                self._json({"status": "ok", "provider": runtime.provider.name})
            elif path == "/api/config":
                self._json({
                    "provider": runtime.provider.name,
                    "model": settings.ollama_model if runtime.provider.name == "ollama" else (
                        settings.model if runtime.provider.name == "openai" else "local-demo-planner"),
                    "tools": [
                        {"name": spec["name"], "description": spec["description"]}
                        for spec in runtime.tools.specs()
                    ],
                })
            elif path == "/api/sessions":
                self._json({"sessions": runtime.store.list_sessions()})
            elif path == "/api/tasks":
                self._json({"tasks": runtime.store.list_tasks()})
            elif path.startswith("/api/runs/"):
                run = runtime.store.get_run(path.rsplit("/", 1)[-1])
                self._json(run or {"error": "run not found"}, HTTPStatus.OK if run else HTTPStatus.NOT_FOUND)
            else:
                self._static(path)

        def do_POST(self) -> None:  # noqa: N802
            path = urlparse(self.path).path
            try:
                body = self._body()
                if path == "/api/chat":
                    session_id = str(body.get("session_id") or new_id("session"))
                    reply = runtime.chat(
                        session_id=session_id,
                        message=str(body.get("message") or ""),
                        user_id=str(body.get("user_id") or "local-user"),
                    )
                    self._json(reply.to_dict())
                elif path.startswith("/api/approvals/"):
                    approval_id = path.rsplit("/", 1)[-1]
                    if type(body.get("approved")) is not bool:
                        raise ValueError("approved must be a boolean")
                    self._json(runtime.approve(approval_id, body["approved"]).to_dict())
                else:
                    self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)
            except (ValueError, KeyError) as exc:
                self._json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
            except Exception as exc:  # defensive API boundary
                self._json({"error": f"internal error: {exc}"}, HTTPStatus.INTERNAL_SERVER_ERROR)

        def _body(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length", "0"))
            if length > 1_000_000:
                raise ValueError("request body is too large")
            raw = self.rfile.read(length)
            try:
                value = json.loads(raw or b"{}")
            except json.JSONDecodeError as exc:
                raise ValueError("invalid JSON body") from exc
            if not isinstance(value, dict):
                raise ValueError("JSON body must be an object")
            return value

        def _json(self, payload: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def _static(self, path: str) -> None:
            relative = "index.html" if path in {"", "/"} else path.lstrip("/")
            candidate = (WEB_ROOT / relative).resolve()
            if WEB_ROOT.resolve() not in candidate.parents and candidate != WEB_ROOT.resolve():
                self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)
                return
            if not candidate.is_file():
                candidate = WEB_ROOT / "index.html"
            data = candidate.read_bytes()
            content_type = mimetypes.guess_type(candidate.name)[0] or "application/octet-stream"
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", f"{content_type}; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, format: str, *args: Any) -> None:
            sys.stdout.write(f"[http] {self.address_string()} {format % args}\n")

    return Handler


def main() -> None:
    try:
        settings = Settings()
        runtime = build_runtime(settings)
    except ValueError as exc:
        raise SystemExit(f"Configuration error: {exc}") from exc
    server = ThreadingHTTPServer((settings.host, settings.port), make_handler(runtime, settings))
    print(f"Native Agent running at http://{settings.host}:{settings.port}")
    print(f"Provider: {runtime.provider.name}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping…")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
