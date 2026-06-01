# business-plan-maker

한 줄짜리 사업 아이디어를 검증·대화를 거쳐 실행 가능한 계획서로 바꿔주는 멀티 에이전트 시스템 (기획서 v0.7.5).

- **백엔드** — Python / FastAPI + LangGraph 오케스트레이터 (`/agents`, `/api_server`)
- **프론트** — Vite + React (`/web`)
- **인프라** — Chroma 벡터 DB 인덱싱 (`/infra`)

---

## 0. 요구사항

| 도구 | 버전 |
|---|---|
| Python | 3.11+ |
| Node.js | 18+ (Vite 5) |
| OpenAI API 키 | **필수** — 없으면 실행되지 않음 |

---

## 1. 백엔드 (api_server)

### 1-1. 가상환경 + 의존성

```bash
cd business-plan-maker
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 1-2. 환경변수

`.env.example`를 복사해서 `.env`로 둔다.

```bash
cp .env.example .env
```

| 변수 | 설명 |
|---|---|
| `OPENAI_API_KEY` | **필수.** 없으면 서버가 기동되지 않는다(mock/키리스 경로 없음). |
| `BPM_LLM_MODEL` | 오케스트레이터 모델 ID (예: `gpt-5-mini` / `gpt-4o-mini` / `gpt-4.1-mini`). |
| `OPENAI_MODEL` | 리서치/RAG 에이전트 모델(Responses API). 미설정 시 `gpt-5.4-mini`. |
| `TAVILY_API_KEY` | 리서치 웹 검색용. 없으면 리서치는 폴백 처리된다. |
| `DOCS_BASE_PATH` / `DIRECTORY_MAP_PATH` | 회사 RAG 벡터DB 경로(`vector_db_store/`). |

> **키가 필요하다** — 이 앱은 mock/키리스 경로가 없다. `OPENAI_API_KEY`가 없으면
> `api_server`가 기동 시점에 거부하고, 그래프 LLM 호출도 즉시 실패한다. `.env`는 서버
> 진입점(`api_server/main.py`)에서 자동 로드된다 — 키를 `.env`에 넣으면 바로 적용된다.

### 1-3. 서버 실행

```bash
source .venv/bin/activate
uvicorn api_server.main:app --reload --port 8000
```

확인:

```bash
curl http://127.0.0.1:8000/health   # {"status":"ok"}
```

주요 엔드포인트: `POST /session`, `POST /chat` (SSE), `POST /plan`, `GET /plan/{id}/download`.

---

## 2. 프론트엔드 (web)

별도 터미널에서:

```bash
cd web
npm install
npm run dev          # http://localhost:5173
```

Vite dev 서버가 `/api/*` 요청을 `http://127.0.0.1:8000`으로 프록시한다 (`vite.config.ts`).
따라서 **백엔드(8000)를 먼저 띄운 뒤 프론트(5173)** 를 실행하면 된다.

프로덕션 빌드:

```bash
npm run build && npm run preview
```

---

## 3. 벡터 DB 인덱싱 (선택, RAG용)

회사 문서 RAG를 쓰려면 PDF를 Chroma DB로 인덱싱한다. OpenAI 임베딩 키가 필요하다.
런타임이 읽는 데이터는 `.env`의 `DOCS_BASE_PATH`(기본 `vector_db_store/documents`)에 있다.

```bash
source .venv/bin/activate
python vector_db_store/src/build_chroma_db.py
```

`documents/` 하위 PDF를 읽어 `chroma_db/`를 생성한다. 파일·설정이 동일하면 재생성을 건너뛴다.
(`vector_db_store/`는 gitignore된 로컬 데이터다.)

---

## 4. 테스트

```bash
source .venv/bin/activate
pytest tests/
```

오케스트레이터 그래프 구성만 빠르게 확인(키 불필요 — 빌드만):

```bash
python -c "from agents.orchestrator.graph import build_graph; build_graph(); print('graph OK')"
```

---

## 5. 그래프 시각화·디버깅 (LangGraph Studio · 선택)

오케스트레이터 그래프를 브라우저에서 시각화하고 노드 단위로 스텝 디버깅한다.
앱 코드 변경 없이 dev 서버만 띄우면 된다 (`langgraph.json`이 그래프를 연결).

```bash
source .venv/bin/activate            # Python 3.11+ 필요 (.venv)
pip install -r requirements-dev.txt  # langgraph-cli[inmem] — 최초 1회
langgraph dev                        # 프로젝트 루트에서
```

→ `http://127.0.0.1:2024` 기동 + Studio가 브라우저로 자동 오픈:
`https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024`

Studio에서 `orchestrator` 그래프를 선택하면 토폴로지
(`confirm_resolve → segment → classify → correction →[조건분기]→ dispatch/gate → … → END`)가 보이고,
노드를 클릭해 입·출력 state를 단계별로 확인할 수 있다. 새 스레드에 아래 입력을 넣어 한 턴을 실행한다:

```json
{ "user_input": "수학 학원용 출석 앱 만들래", "turn": 1, "slots": {}, "messages": [],
  "turn_segments": [], "turn_validation_reports": [], "pending_clarifications": [] }
```

> **참고**
> - Studio에서 노드를 실제로 실행하려면 `OPENAI_API_KEY`가 필요하다(mock 경로 없음).
> - `BlockingError`가 나면 `langgraph dev --allow-blocking`으로 실행한다.
> - Studio는 컴파일된 그래프를 직접 실행하므로 `run_turn`의 턴 전처리(turn 증가·메시지 적재·
>   턴-로컬 리셋)는 적용되지 않는다 — 단일 턴 노드 디버깅용.

---

## 6. 디렉토리 한눈에

```
/agents          오케스트레이터 + 워커 (conversation·research·rag·logic_validator·validator·planner)
/api_server      FastAPI 진입점 · SSE · 세션 · PDF 렌더
/common          공유 스키마·베이스 클래스
/infra           벡터 DB·인덱싱 파이프라인
/web             Vite + React 프론트
/docs            코드 워크스루 문서
```

---

## 빠른 시작 (요약)

```bash
# 터미널 1 — 백엔드
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env            # OPENAI_API_KEY 필수 (키 없으면 실행 안 됨)
uvicorn api_server.main:app --reload --port 8000

# 터미널 2 — 프론트
cd web && npm install && npm run dev
```

브라우저에서 http://localhost:5173 접속.
