# LangChain Agent

## LangChain + LlamaIndex 기반 Agent 실행

실제 LangChain `create_agent` / LangGraph 실행 루프를 사용하는 모드를 추가했습니다.
문서 검색은 LlamaIndex의 Markdown 파서와 BM25 인덱스를 사용합니다.
설치·실행·설계·검증 안내는 [LangChain Agent 가이드](docs/langchain-agent.md)를 참고하세요.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[langchain]'
python -m agent_langchain '휴가 정책을 찾고 128 * 42를 계산해줘' --json
```

기본 데모는 규칙 기반 모델이며, 계산과 검색 도구는 LangChain을 통해 실제 실행됩니다.
자연어로 도구를 선택·조합하는 실제 LLM 실행은 `--provider openai`를 사용합니다.

## Ollama로 Operations Copilot 화면 실행

```bash
AGENT_PROVIDER=ollama OLLAMA_MODEL=qwen3:latest AGENT_PORT=8091 python -m agent_native
```

Ollama가 실행 중이면 `http://127.0.0.1:8091`에서 로컬 AI를 사용할 수 있습니다.
LangChain ChatOllama + LlamaIndex 검색을 연결했으며, 도구·승인·메모리·추적을 실제 모델로 검증했습니다.

기업 환경에서 확장할 수 있는 **에이전트 런타임 MVP**입니다. 모델이 단순히 답변만 생성하는 대신 도구를 선택하고 실행하며, 부작용이 있는 작업은 사람의 승인을 받은 뒤 이어서 처리합니다.

## 포함된 기능

- OpenAI Responses API 어댑터와 API 키 없이 실행되는 데모 모델
- 함수 도구 호출 루프와 단계 수 제한
- 읽기 도구 자동 실행, 쓰기 도구 승인(Human-in-the-loop)
- 세션 메모리, 사용자 메모, 간단한 로컬 지식 검색
- 실행 이벤트/지연 시간 추적과 회귀 평가 CLI
- 의존성 없는 Python HTTP API와 반응형 웹 콘솔
- 모델 공급자와 도구를 분리한 교체 가능한 구조

## 빠른 시작

Python 3.11 이상만 필요합니다.

```bash
python3 -m agent_native
```

브라우저에서 `http://127.0.0.1:8080`을 엽니다. 기본값은 `demo` 공급자이므로 API 키가 필요 없습니다.

실제 모델을 사용하려면:

```bash
export AGENT_PROVIDER=openai
export OPENAI_API_KEY=your_api_key
export OPENAI_MODEL=gpt-5.4-mini
python3 -m agent_native
```

환경 변수 전체는 [.env.example](.env.example)을 참고하세요.

## 데모 프롬프트

- `128 * 42를 계산해줘`
- `휴가 정책을 찾아줘`
- `내 이름은 민수라고 기억해줘`
- `금요일까지 보안 리뷰 할 일을 만들어줘` — 승인 카드가 표시됩니다.

## API

```text
GET  /api/health
GET  /api/config
GET  /api/sessions
GET  /api/tasks
GET  /api/runs/{run_id}
POST /api/chat                 {"session_id":"...", "message":"..."}
POST /api/approvals/{id}       {"approved":true}
```

## 검증

```bash
python3 -m unittest discover -s tests -v
python3 -m agent_native.evaluation
```

## 구조

```text
Browser UI -> HTTP API -> AgentRuntime -> ModelProvider
                            |      |
                            |      +-> ToolRegistry -> Knowledge / Tasks / Memory
                            +-> StateStore -> Sessions / Approvals / Traces
```

현재 저장소는 로컬 JSON을 사용합니다. 운영 환경에서는 `StateStore` 인터페이스 뒤에 PostgreSQL/Redis를, `KnowledgeSearchTool` 뒤에 사내 검색 또는 벡터 DB를 연결하고 인증·테넌트 분리·감사 로그를 추가하는 구성이 적합합니다.
