"""Live local-model checks. Run explicitly; uses disposable application data."""

import tempfile
from pathlib import Path

from agent_native.api import build_runtime
from agent_native.config import Settings
from .runtime import ask, build_agent


def main():
    with tempfile.TemporaryDirectory() as directory:
        settings = Settings(provider="ollama", data_dir=Path(directory))
        runtime = build_runtime(settings)

        def completed(reply):
            assert reply.status == "completed", (reply.status, reply.message)

        reply = runtime.chat("math", "128 * 42를 calculator 도구로 계산해줘")
        completed(reply)
        assert "5376" in reply.message or "5,376" in reply.message, reply.message
        assert any(e["type"] == "tool_result" for e in runtime.store.get_run(reply.run_id)["events"])
        print("PASS 계산 + 실행 추적", flush=True)

        reply = runtime.chat("policy", "휴가 정책을 검색하고 출처를 알려줘")
        completed(reply)
        assert "3" in reply.message and "employee-handbook.md" in reply.message, reply.message
        print("PASS LlamaIndex 문서 검색 + 출처", flush=True)

        reply = runtime.chat("memory", "내 이름은 테스트민수야. name 키로 기억해줘")
        completed(reply)
        assert runtime.store.memories("local-user").get("name") == "테스트민수"
        runtime = build_runtime(settings)
        reply = runtime.chat("recall-new-session", "내 이름은 뭐야?")
        completed(reply)
        assert "테스트민수" in reply.message, reply.message
        print("PASS 메모리 저장 + 재시작 후 새 대화에서 기억", flush=True)

        reply = runtime.chat("approve", "도구를 호출해서 'Ollama 테스트 보안 검토' 할 일을 만들어줘. 마감일은 없어.")
        assert reply.status == "approval_required", reply.message
        assert not runtime.store.list_tasks()
        done = runtime.approve(reply.approval["id"], True)
        completed(done)
        assert len(runtime.store.list_tasks()) == 1
        print("PASS 승인 전 실행 차단 + 승인 후 생성", flush=True)

        reply = runtime.chat("deny", "'Ollama 거절 테스트' 할 일을 만들어줘. 마감일은 없어.")
        assert reply.status == "approval_required", reply.message
        done = runtime.approve(reply.approval["id"], False)
        completed(done)
        assert len(runtime.store.list_tasks()) == 1
        print("PASS 거절 시 생성 안 함", flush=True)

        agent = build_agent(provider="ollama")
        result = ask(agent, "휴가 정책을 찾아 출처를 표시하고 128 * 42를 계산 도구로 계산해줘")
        names = {e.get("name") for e in result["trace"] if e["type"] == "tool_result"}
        assert names == {"calculator", "search_knowledge"}, result
        assert "5376" in str(result["answer"]) or "5,376" in str(result["answer"]), result
        print("PASS LangChain create_agent + Ollama + LlamaIndex 복합 요청", flush=True)
        print("6/6 live Ollama checks passed", flush=True)


if __name__ == "__main__":
    main()
