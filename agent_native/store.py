from __future__ import annotations

import json
import threading
from copy import deepcopy
from pathlib import Path
from typing import Any

from .models import now_iso


class JsonStateStore:
    """Small durable store. Replace behind this API for multi-instance deployments."""

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._state: dict[str, Any] = {
            "sessions": {},
            "approvals": {},
            "runs": {},
            "tasks": [],
            "memories": {},
        }
        if self.path.exists():
            try:
                loaded = json.loads(self.path.read_text(encoding="utf-8"))
                for key in self._state:
                    self._state[key] = loaded.get(key, self._state[key])
            except (json.JSONDecodeError, OSError):
                pass

    def _flush(self) -> None:
        temp = self.path.with_suffix(".tmp")
        temp.write_text(json.dumps(self._state, ensure_ascii=False, indent=2), encoding="utf-8")
        temp.replace(self.path)

    def get_session(self, session_id: str) -> dict[str, Any]:
        with self._lock:
            session = self._state["sessions"].setdefault(
                session_id,
                {"id": session_id, "user_id": "local-user", "items": [], "updated_at": now_iso()},
            )
            return deepcopy(session)

    def save_session(self, session: dict[str, Any]) -> None:
        with self._lock:
            session["updated_at"] = now_iso()
            self._state["sessions"][session["id"]] = deepcopy(session)
            self._flush()

    def list_sessions(self) -> list[dict[str, Any]]:
        with self._lock:
            result = []
            for session in self._state["sessions"].values():
                messages = [i for i in session["items"] if i.get("role") == "user"]
                result.append({
                    "id": session["id"],
                    "title": (messages[0].get("content", "새 대화") if messages else "새 대화")[:40],
                    "updated_at": session["updated_at"],
                })
            return sorted(result, key=lambda item: item["updated_at"], reverse=True)

    def put_approval(self, approval: dict[str, Any]) -> None:
        with self._lock:
            self._state["approvals"][approval["id"]] = deepcopy(approval)
            self._flush()

    def get_approval(self, approval_id: str) -> dict[str, Any] | None:
        with self._lock:
            value = self._state["approvals"].get(approval_id)
            return deepcopy(value) if value else None

    def update_approval(self, approval_id: str, updates: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            approval = self._state["approvals"][approval_id]
            approval.update(updates)
            self._flush()
            return deepcopy(approval)

    def start_run(self, run: dict[str, Any]) -> None:
        with self._lock:
            self._state["runs"][run["id"]] = deepcopy(run)
            self._flush()

    def add_event(self, run_id: str, event: dict[str, Any]) -> None:
        with self._lock:
            self._state["runs"][run_id]["events"].append(deepcopy(event))
            self._flush()

    def finish_run(self, run_id: str, status: str) -> None:
        with self._lock:
            run = self._state["runs"][run_id]
            run["status"] = status
            run["finished_at"] = now_iso()
            self._flush()

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        with self._lock:
            value = self._state["runs"].get(run_id)
            return deepcopy(value) if value else None

    def add_task(self, task: dict[str, Any]) -> None:
        with self._lock:
            self._state["tasks"].append(deepcopy(task))
            self._flush()

    def list_tasks(self) -> list[dict[str, Any]]:
        with self._lock:
            return deepcopy(self._state["tasks"])

    def remember(self, user_id: str, key: str, value: str) -> None:
        with self._lock:
            self._state["memories"].setdefault(user_id, {})[key] = value
            self._flush()

    def memories(self, user_id: str) -> dict[str, str]:
        with self._lock:
            return deepcopy(self._state["memories"].get(user_id, {}))

