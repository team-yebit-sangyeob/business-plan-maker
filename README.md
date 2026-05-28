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
| (선택) OpenAI API 키 | LLM live 모드용 |

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
| `OPENAI_API_KEY` | OpenAI 키. 비워두면 자동으로 mock 모드. |
| `BPM_LLM_MODE` | `live` 또는 `mock`. `mock`이면 OpenAI 호출 없이 등록된 mock 핸들러 사용. |
| `BPM_LLM_MODEL` | 모델 ID (예: `gpt-5-mini` / `gpt-4o-mini` / `gpt-4.1-mini`). |

> **키 없이 돌려보기** — `BPM_LLM_MODE=mock`이면 OpenAI 호출 없이 빌트인 데모 핸들러
> (`agents/orchestrator/mocks.py`)가 그럴듯한 한국어 응답·슬롯 채움·게이트 흐름을 만들어
> 준다. UI 흐름 점검·오프라인 데모용. 실제 판단 품질은 `live` 모드에서만.
>
> `.env`는 서버 진입점(`api_server/main.py`)에서 자동 로드된다 — 키를 `.env`에 넣고
> `BPM_LLM_MODE=live`로 두면 별도 export 없이 바로 적용된다.

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

```bash
source .venv/bin/activate
python infra/vector_db/src/build_chroma_db.py
```

`documents/` 하위 PDF를 읽어 `chroma_db/`를 생성한다. 파일·설정이 동일하면 재생성을 건너뛴다.

---

## 4. 테스트

```bash
source .venv/bin/activate
BPM_LLM_MODE=mock pytest tests/
```

오케스트레이터 그래프만 빠르게 확인:

```bash
BPM_LLM_MODE=mock python -c "from agents.orchestrator.graph import build_graph; build_graph(); print('graph OK')"
```

---

## 5. 디렉토리 한눈에

```
/agents          오케스트레이터 + 워커 (conversation·research·rag·critic·planner)
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
cp .env.example .env            # 키 없으면 BPM_LLM_MODE=mock
uvicorn api_server.main:app --reload --port 8000

# 터미널 2 — 프론트
cd web && npm install && npm run dev
```

브라우저에서 http://localhost:5173 접속.
