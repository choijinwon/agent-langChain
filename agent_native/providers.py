from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from typing import Any

from .models import ModelTurn, ToolCall, new_id


class ProviderError(RuntimeError):
    pass


class ModelProvider(ABC):
    name = "base"

    @abstractmethod
    def generate(
        self,
        items: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        instructions: str,
    ) -> ModelTurn:
        raise NotImplementedError


class DemoProvider(ModelProvider):
    """Deterministic local planner for exploring the runtime without credentials."""

    name = "demo"

    def generate(self, items, tools, instructions) -> ModelTurn:  # noqa: ARG002
        if items and items[-1].get("type") == "function_call_output":
            raw = items[-1].get("output", "{}")
            try:
                result = json.loads(raw)
            except json.JSONDecodeError:
                result = {"result": raw}
            if "result" in result:
                text = f"계산 결과는 **{result['result']}**입니다."
            elif result.get("matches"):
                text = "사내 지식에서 다음 내용을 찾았습니다.\n\n" + "\n\n".join(
                    match["content"] for match in result["matches"][:2]
                )
            elif result.get("saved"):
                text = f"기억해 두었습니다: **{result['key']} = {result['value']}**"
            elif result.get("created"):
                text = f"할 일을 만들었습니다: **{result['task']['title']}**"
            elif result.get("denied"):
                text = "요청한 작업은 승인되지 않아 실행하지 않았습니다."
            else:
                text = f"도구 실행을 마쳤습니다. `{json.dumps(result, ensure_ascii=False)}`"
            return self._text_turn(text)

        user_messages = [item for item in items if item.get("role") == "user"]
        message = str(user_messages[-1].get("content", "")) if user_messages else ""
        lowered = message.lower()

        # Korean particles are Unicode ``\w`` characters, so word-boundary lookarounds
        # would truncate a number immediately followed by "를"/"은".
        expression_match = re.search(r"[\d\s+\-*/().%]{3,}", message)
        if expression_match and ("계산" in message or any(op in expression_match.group(1) for op in "*/+")):
            return self._tool_turn("calculator", {"expression": expression_match.group(0).strip()})
        if any(word in message for word in ("휴가", "보안 정책", "비용 정책", "사내 정책")):
            return self._tool_turn("search_knowledge", {"query": message})
        if "기억" in message:
            fact = message.replace("기억해줘", "").replace("기억해 줘", "").strip(" .")
            if "이름" in fact:
                match = re.search(r"이름(?:은|이)?\s*([^\s]+)", fact)
                value = match.group(1).rstrip("라고") if match else fact
                return self._tool_turn("remember_user_fact", {"key": "name", "value": value})
            return self._tool_turn("remember_user_fact", {"key": "note", "value": fact})
        if any(word in lowered for word in ("할 일", "task", "todo", "업무를 만들어")):
            title = re.sub(r"(할 일을?|task|todo)\s*(만들어줘|추가해줘|생성해줘)?", "", message, flags=re.I)
            title = title.strip(" .") or message
            due = "금요일" if "금요일" in message else None
            return self._tool_turn("create_task", {"title": title, "due": due})

        memories = ""
        if "[사용자 메모]" in instructions:
            memories = " 이전에 저장된 사용자 정보도 문맥에 반영할 수 있습니다."
        return self._text_turn(
            "안녕하세요. 저는 도구 호출, 승인, 메모리와 실행 추적을 갖춘 Native Agent입니다. "
            "계산, 사내 정책 검색, 사용자 정보 기억, 승인형 할 일 생성을 요청해 보세요." + memories
        )

    @staticmethod
    def _text_turn(text: str) -> ModelTurn:
        item = {"type": "message", "role": "assistant", "content": text}
        return ModelTurn(text=text, output_items=[item])

    @staticmethod
    def _tool_turn(name: str, arguments: dict[str, Any]) -> ModelTurn:
        call_id = new_id("call")
        item = {
            "type": "function_call",
            "call_id": call_id,
            "name": name,
            "arguments": json.dumps(arguments, ensure_ascii=False),
        }
        return ModelTurn(
            tool_calls=[ToolCall(call_id=call_id, name=name, arguments=arguments)],
            output_items=[item],
        )


class OpenAIResponsesProvider(ModelProvider):
    name = "openai"

    def __init__(self, api_key: str, model: str, base_url: str):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url

    def generate(self, items, tools, instructions) -> ModelTurn:
        body = {
            "model": self.model,
            "instructions": instructions,
            "input": items,
            "tools": tools,
            "tool_choice": "auto",
            "parallel_tool_calls": False,
            "include": ["reasoning.encrypted_content"],
            "store": False,
        }
        request = urllib.request.Request(
            f"{self.base_url}/responses",
            data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise ProviderError(f"OpenAI API error {exc.code}: {detail[:500]}") from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            raise ProviderError(f"OpenAI API connection failed: {exc}") from exc

        text_parts: list[str] = []
        calls: list[ToolCall] = []
        output_items = payload.get("output", [])
        for item in output_items:
            if item.get("type") == "function_call":
                try:
                    arguments = json.loads(item.get("arguments") or "{}")
                except json.JSONDecodeError as exc:
                    raise ProviderError(f"Invalid tool arguments for {item.get('name')}") from exc
                calls.append(ToolCall(
                    call_id=item["call_id"],
                    name=item["name"],
                    arguments=arguments,
                ))
            elif item.get("type") == "message":
                for content in item.get("content", []):
                    if content.get("type") == "output_text":
                        text_parts.append(content.get("text", ""))
        usage = payload.get("usage") or {}
        return ModelTurn(
            text="\n".join(text_parts).strip(),
            tool_calls=calls,
            output_items=output_items,
            usage={key: int(value) for key, value in usage.items() if isinstance(value, int)},
        )
