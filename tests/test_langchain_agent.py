import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

HAS_LANGCHAIN = importlib.util.find_spec("langchain") is not None
if HAS_LANGCHAIN:
    from agent_langchain.runtime import ask, build_agent
    from agent_langchain.tools import build_tools
    from agent_langchain.demo import DemoChatModel
    from langchain_core.messages import AIMessage, HumanMessage
    from langchain_core.outputs import ChatGeneration, ChatResult
    from langgraph.errors import GraphRecursionError


@unittest.skipUnless(HAS_LANGCHAIN, "Install .[langchain] to run framework integration tests")
class LangChainAgentTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        (self.root / "policy.md").write_text("## 휴가 정책\n휴가는 3영업일 전에 신청합니다.", encoding="utf-8")
        self.agent = build_agent(knowledge_dir=self.root)

    def tearDown(self):
        self.directory.cleanup()

    def test_compound_request_executes_real_tools(self):
        reply = ask(self.agent, "휴가 정책을 찾고 128 * 42를 계산해줘")
        self.assertIn("5376", reply["answer"])
        self.assertIn("[policy.md]", reply["answer"])
        self.assertIn("3영업일", reply["answer"])
        results = [event for event in reply["trace"] if event["type"] == "tool_result"]
        self.assertEqual({event["name"] for event in results}, {"calculator", "search_knowledge"})
        self.assertEqual(len(results), 2)

    def test_tool_failure_returns_observation(self):
        reply = ask(self.agent, "1 / 0을 계산해줘")
        self.assertIn("도구 오류", reply["answer"])
        self.assertIn("error", json.loads(reply["trace"][-1]["content"]))

    def test_calculator_rejects_code_and_expensive_expressions(self):
        calculator = build_tools(self.root)[0]
        for expression in ("__import__('os').getcwd()", "2**1000000", "1e309", "1" * 101):
            with self.subTest(expression=expression):
                self.assertIn("error", calculator.invoke({"expression": expression}))

    def test_no_results_is_explicit(self):
        reply = ask(self.agent, "없는문서 검색")
        self.assertIn("찾지 못했습니다", reply["answer"])

    def test_search_does_not_follow_symlink(self):
        with tempfile.TemporaryDirectory() as outside:
            secret = Path(outside) / "secret.md"
            secret.write_text("휴가 secret", encoding="utf-8")
            (self.root / "external.md").symlink_to(secret)
            result = build_tools(self.root)[1].invoke({"query": "secret"})
            self.assertEqual(result["matches"], [])

    def test_sessions_remember_and_are_isolated(self):
        class HistoryModel(DemoChatModel):
            def _generate(self, messages, **kwargs):
                content = "|".join(m.content for m in messages if isinstance(m, HumanMessage))
                return ChatResult(generations=[ChatGeneration(message=AIMessage(content=content))])
        agent = build_agent(chat_model=HistoryModel(), knowledge_dir=self.root)
        ask(agent, "첫 메시지", "a")
        self.assertEqual(ask(agent, "두 번째", "a")["answer"], "첫 메시지|두 번째")
        self.assertEqual(ask(agent, "별도", "b")["answer"], "별도")

    def test_trace_does_not_repeat_previous_turn(self):
        ask(self.agent, "2 + 2 계산")
        reply = ask(self.agent, "3 + 3 계산")
        self.assertEqual(len(reply["trace"]), 2)
        self.assertIn("6", reply["answer"])

    def test_endless_model_stops_at_budget(self):
        class LoopModel(DemoChatModel):
            def _generate(self, messages, **kwargs):
                message = AIMessage(content="", tool_calls=[{
                    "name": "calculator", "args": {"expression": "1+1"},
                    "id": f"call-{len(messages)}", "type": "tool_call",
                }])
                return ChatResult(generations=[ChatGeneration(message=message)])
        agent = build_agent(chat_model=LoopModel(), knowledge_dir=self.root)
        with self.assertRaises(GraphRecursionError):
            ask(agent, "계속", max_steps=4)

    def test_missing_credentials_and_invalid_input(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, "OPENAI_API_KEY"):
                build_agent("openai")
        with self.assertRaises(ValueError):
            ask(self.agent, " ")
        with self.assertRaises(ValueError):
            ask(self.agent, "안녕", max_steps=0)
