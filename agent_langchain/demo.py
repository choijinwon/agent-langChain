"""Deterministic test/demo model, not an LLM. Tools still run through LangChain."""

import json
import re
from uuid import uuid4

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult


class DemoChatModel(BaseChatModel):
    @property
    def _llm_type(self):
        return "local-deterministic-demo"

    def bind_tools(self, tools, *, tool_choice=None, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        if isinstance(messages[-1], ToolMessage):
            results = []
            for message in reversed(messages):
                if not isinstance(message, ToolMessage):
                    break
                result = json.loads(message.content)
                if "error" in result:
                    results.append(f"도구 오류: {result['error']}")
                elif "result" in result:
                    results.append(f"계산 결과: {result['result']}")
                else:
                    results.append("\n".join(
                        f"[{item['source']}] {item['content']}"
                        for item in result.get("matches", [])
                    ) or "관련 문서를 찾지 못했습니다.")
            answer = AIMessage(content="\n\n".join(reversed(results)))
        else:
            message = next(m.content for m in reversed(messages) if isinstance(m, HumanMessage))
            calls = []
            expression = re.search(r"\d[\d\s+*/().%\-]*", message)
            if expression and ("계산" in message or any(c in expression[0] for c in "+-*/%")):
                calls.append({"name": "calculator", "args": {"expression": expression[0].strip()},
                              "id": uuid4().hex, "type": "tool_call"})
            if any(word in message for word in ("정책", "휴가", "검색", "찾아")):
                query = " ".join(word for word in ("휴가", "보안", "비용") if word in message) or message
                calls.append({"name": "search_knowledge", "args": {"query": query},
                              "id": uuid4().hex, "type": "tool_call"})
            answer = AIMessage(content="" if calls else
                               "데모는 계산과 정책 검색을 지원합니다. 자유로운 요청은 openai 모드를 사용하세요.",
                               tool_calls=calls)
        return ChatResult(generations=[ChatGeneration(message=answer)])
