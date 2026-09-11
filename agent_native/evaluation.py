from __future__ import annotations

import tempfile
from pathlib import Path

from .providers import DemoProvider
from .runtime import AgentRuntime
from .store import JsonStateStore
from .tools import ToolRegistry


CASES = [
    ("12 * 9를 계산해줘", "108", "completed"),
    ("휴가 정책을 찾아줘", "3영업일", "completed"),
    ("보안 검토 할 일을 만들어줘", "승인", "approval_required"),
]


def main() -> int:
    project_root = Path(__file__).resolve().parent.parent
    passed = 0
    with tempfile.TemporaryDirectory() as directory:
        store = JsonStateStore(Path(directory) / "state.json")
        runtime = AgentRuntime(DemoProvider(), ToolRegistry(store, project_root / "knowledge"), store)
        for index, (prompt, expected, status) in enumerate(CASES, start=1):
            reply = runtime.chat(f"eval-{index}", prompt)
            haystack = f"{reply.message} {reply.approval or ''}"
            ok = reply.status == status and expected in haystack
            passed += int(ok)
            print(f"{'PASS' if ok else 'FAIL'}  {prompt} -> {reply.status}")
    print(f"\n{passed}/{len(CASES)} cases passed")
    return 0 if passed == len(CASES) else 1


if __name__ == "__main__":
    raise SystemExit(main())

