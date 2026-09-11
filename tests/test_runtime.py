from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from agent_native.providers import DemoProvider
from agent_native.runtime import AgentRuntime
from agent_native.store import JsonStateStore
from agent_native.tools import ToolError, ToolRegistry


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.store = JsonStateStore(Path(self.temp_dir.name) / "state.json")
        knowledge = Path(__file__).resolve().parent.parent / "knowledge"
        tools = ToolRegistry(self.store, knowledge)
        self.runtime = AgentRuntime(DemoProvider(), tools, self.store)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_calculator_tool_loop(self):
        reply = self.runtime.chat("math", "128 * 42를 계산해줘")
        self.assertEqual(reply.status, "completed")
        self.assertIn("5376", reply.message)
        run = self.store.get_run(reply.run_id)
        self.assertTrue(any(event["type"] == "tool_result" for event in run["events"]))

    def test_knowledge_search(self):
        reply = self.runtime.chat("policy", "휴가 정책을 찾아줘")
        self.assertEqual(reply.status, "completed")
        self.assertIn("3영업일", reply.message)

    def test_external_write_requires_approval(self):
        reply = self.runtime.chat("task", "보안 검토 할 일을 만들어줘")
        self.assertEqual(reply.status, "approval_required")
        self.assertEqual(self.store.list_tasks(), [])
        completed = self.runtime.approve(reply.approval["id"], True)
        self.assertEqual(completed.status, "completed")
        self.assertEqual(len(self.store.list_tasks()), 1)

    def test_denied_action_has_no_side_effect(self):
        reply = self.runtime.chat("deny", "보고서 작성 할 일을 만들어줘")
        completed = self.runtime.approve(reply.approval["id"], False)
        self.assertIn("승인되지 않아", completed.message)
        self.assertEqual(self.store.list_tasks(), [])

    def test_memory_is_persisted(self):
        reply = self.runtime.chat("memory", "내 이름은 민수라고 기억해줘")
        self.assertEqual(reply.status, "completed")
        self.assertIn("name", self.store.memories("local-user"))

    def test_calculator_rejects_code(self):
        with self.assertRaises(ToolError):
            self.runtime.tools.execute(
                "calculator", {"expression": "__import__('os').system('id')"}, {"user_id": "u"}
            )


if __name__ == "__main__":
    unittest.main()

