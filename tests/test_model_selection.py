import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from agent_native.api import build_runtime
from agent_native.config import Settings
from agent_native.model_selection import ModelRouter


class ModelSelectionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.settings = Settings(provider='demo', data_dir=Path(self.temp.name))
        self.runtime = build_runtime(self.settings)
        self.router = ModelRouter(self.runtime, self.settings)

    def tearDown(self):
        self.temp.cleanup()

    def test_unknown_model_rejected_before_session_write(self):
        with self.assertRaises(ValueError):
            self.router.chat('s', 'test', 'u', 'nonexistent')
        self.assertEqual(self.runtime.store.list_sessions(), [])

    def test_request_routes_to_selected_model_without_mutating_default(self):
        choices = [{'name': name, 'available': True} for name in ['model-a', 'model-b']]
        with patch('agent_native.model_selection.list_models', return_value=choices), patch.object(self.router, 'selected_runtime', return_value=self.runtime) as factory:
            self.router.chat('a', '안녕', 'u', 'model-a')
            self.router.chat('b', '안녕', 'u', 'model-b')
            self.assertEqual([c.args[0] for c in factory.call_args_list], ['model-a', 'model-b'])
            self.assertEqual(self.runtime.provider.name, 'demo')

    def test_approval_uses_original_model_and_blocks_new_chat(self):
        self.runtime.provider.model = 'original-model'
        reply = self.router.chat('a', '보안 검토 할 일을 만들어줘', 'u')
        with self.assertRaisesRegex(ValueError, '이전 작업'):
            self.router.chat('a', '다른 메시지', 'u')
        with patch.object(self.router, 'selected_runtime', return_value=self.runtime) as factory:
            self.router.approve(reply.approval['id'], True)
            factory.assert_called_once_with('original-model')
        self.assertEqual(len(self.runtime.store.list_tasks()), 1)

    def test_unsupported_model_cannot_be_selected(self):
        with patch('agent_native.model_selection.list_models', return_value=[{'name':'embedding', 'available':False}]):
            with self.assertRaises(ValueError):
                self.router.chat('s', 'test', 'u', 'embedding')
