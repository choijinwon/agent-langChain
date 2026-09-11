"""Installed model discovery and request-scoped model routing."""
import json
import threading
import urllib.request
from .runtime import AgentRuntime


def list_models(settings):
    if settings.provider != "ollama":
        return [{"name": settings.model if settings.provider == "openai" else "local-demo-planner",
                 "available": True, "reason": ""}]
    base = settings.ollama_base_url.rstrip("/")
    try:
        with urllib.request.urlopen(base + "/api/tags", timeout=5) as response:
            installed = json.load(response)["models"]
    except (OSError, ValueError, KeyError) as exc:
        raise ValueError("Ollama 모델 목록을 불러오지 못했습니다. 서버 실행 상태를 확인하세요.") from exc
    result = []
    for model in installed:
        name = model["name"]
        try:
            request = urllib.request.Request(base + "/api/show", data=json.dumps({"model": name}).encode(),
                                             headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(request, timeout=5) as response:
                capabilities = json.load(response).get("capabilities", [])
            enabled = "tools" in capabilities
            reason = "" if enabled else "Agent 도구 호출 미지원"
        except (OSError, ValueError):
            enabled, reason = False, "모델 정보 조회 실패"
        result.append({"name": name, "available": enabled, "reason": reason,
                       "size_gb": round(model.get("size", 0) / 1e9, 1)})
    return result


class ModelRouter:
    def __init__(self, runtime, settings):
        self.runtime, self.settings = runtime, settings
        self.lock = threading.RLock()

    @property
    def default_model(self):
        return self.settings.ollama_model if self.settings.provider == "ollama" else (
            self.settings.model if self.settings.provider == "openai" else "local-demo-planner")

    def selected_runtime(self, model):
        if self.settings.provider != "ollama":
            if model != self.default_model:
                raise ValueError("현재 공급자에서 사용할 수 없는 모델입니다.")
            return self.runtime
        from .ollama_provider import OllamaProvider
        return AgentRuntime(OllamaProvider(model, self.settings.ollama_base_url),
                            self.runtime.tools, self.runtime.store, self.runtime.max_steps)

    def chat(self, session_id, message, user_id, model=None):
        model = model or self.default_model
        if not isinstance(model, str):
            raise ValueError("모델 이름은 문자열이어야 합니다.")
        with self.lock:
            choices = list_models(self.settings)
            if not any(item["name"] == model and item["available"] for item in choices):
                raise ValueError("설치된 도구 호출 지원 모델을 선택하세요. 모델 목록을 새로고침해 주세요.")
            session = self.runtime.store.get_session(session_id)
            answered = {i.get("call_id") for i in session["items"] if i.get("type") == "function_call_output"}
            if any(i.get("type") == "function_call" and i.get("call_id") not in answered for i in session["items"]):
                raise ValueError("먼저 이전 작업의 승인을 처리하거나 새 대화를 시작하세요.")
            return self.selected_runtime(model).chat(session_id, message, user_id)

    def approve(self, approval_id, approved):
        with self.lock:
            approval = self.runtime.store.get_approval(approval_id)
            if not approval:
                raise KeyError("approval not found")
            run = self.runtime.store.get_run(approval["run_id"])
            # Resume with the original model even if the dropdown was changed meanwhile.
            return self.selected_runtime(run.get("model") or self.default_model).approve(approval_id, approved)
