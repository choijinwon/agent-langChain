import argparse
import json
import sys


def main():
    parser = argparse.ArgumentParser(description="LangChain 계산·문서 검색 Agent")
    parser.add_argument("message", nargs="?", help="생략하면 대화 모드")
    parser.add_argument("--provider", choices=["demo", "openai", "ollama"], default="demo")
    parser.add_argument("--model")
    parser.add_argument("--knowledge-dir")
    parser.add_argument("--session", default="default")
    parser.add_argument("--max-steps", type=int, default=12)
    parser.add_argument("--json", action="store_true", help="도구 실행 기록 포함")
    args = parser.parse_args()
    try:
        from .runtime import ask, build_agent
        agent = build_agent(args.provider, args.model, args.knowledge_dir)
        if args.provider == "demo":
            print("데모 모드: 규칙 기반 모델 + 실제 LangChain 도구 실행", file=sys.stderr)

        def run(message):
            result = ask(agent, message, args.session, args.max_steps)
            print(json.dumps(result, ensure_ascii=False, indent=2) if args.json else result["answer"])

        if args.message is not None:
            run(args.message)
        else:
            print("대화를 입력하세요. 종료: /exit (대화 기억은 현재 프로세스 동안 유지)")
            while True:
                message = input("나> ")
                if message.strip() == "/exit":
                    break
                if message.strip():
                    run(message)
    except ImportError as exc:
        print(f"의존성을 설치하세요: python -m pip install -e '.[langchain]' ({exc})", file=sys.stderr)
        raise SystemExit(1)
    except (EOFError, KeyboardInterrupt):
        print()
    except Exception as exc:
        # Avoid dumping provider error bodies, which can include user content.
        detail = str(exc) if isinstance(exc, ValueError) else type(exc).__name__
        print(f"실행 실패: {detail}. 설정·네트워크·실행 단계 제한을 확인하세요.", file=sys.stderr)
        raise SystemExit(1)
