# 오케스트레이터 로직 총정리 (구현 명세)

> 기획서(`agent_plan_v7_5.html`, `orchestrator_spec.html`) v0.7.5 를 코드로 옮긴 결과.
> **기획서를 먼저 읽고 이 문서를 읽으면** 오케스트레이터가 실제로 어떻게 동작하는지
> 노드 단위로 파악할 수 있다. 코드 위치는 모두 `agents/orchestrator/` 기준.

---

## 0. 한 문장 요약

매 사용자 메시지마다 **단일 진입점**으로 들어와, LangGraph 상태머신이
`confirm_resolve → segment → classify → correction → (dispatch → extract_fills) → gate → conversation → integrator`
순서로 흐르며 **(보류된 슬롯 확인 해소)·분류·세그멘테이션·라우팅·슬롯 추적·출력 게이트**를 수행한다.
판단(무엇을 할지)만 오케스트레이터가 하고, 표현(자연어)·실행(검색·추론)은 워커가 맡는다.

---

## 1. 핵심 데이터 모델 (`common/schema/state.py`)

### 슬롯 10개 — 질문(조사) 순서대로

`ALL_SLOTS` 튜플 순서가 곧 **기본 질문 순서**(사업계획 자연 전개): 문제→고객→솔루션→시장→차별점→수익모델→목표→자원→일정→리스크.

| # | 슬롯 | 구분 | 의미 |
|---|---|---|---|
| 1 | `problem` | **필수** | 문제 정의 (P) |
| 2 | `target` | **필수** | 타겟 고객 (T) |
| 3 | `solution` | 선택 | 솔루션 형태 |
| 4 | `market` | 선택 | 시장 근거 (리서치가 채움) |
| 5 | `advantage` | 선택 | 차별점·경쟁우위 — solution+market 뒤라야 나옴 |
| 6 | `revenue` | 선택 | 수익 모델 |
| 7 | `goal` | **필수** | 목표 수치 (G) — 솔루션·수익모델 뒤라야 현실적 숫자 |
| 8 | `resources` | 선택 | 인력·예산 |
| 9 | `milestones` | 선택 | 일정 단계 |
| 10 | `risks` | 선택 | 리스크 |

```python
# 질문 순서 = 자연 전개 순서 (이 튜플 순서대로 첫 빈칸을 묻는다)
ALL_SLOTS = ("problem", "target", "solution", "market", "advantage",
             "revenue", "goal", "resources", "milestones", "risks")
# 출력 게이트 필수 — 순서가 아닌 멤버십(P·T·G). goal은 질문은 7번째지만 출력 전 충족 강제.
REQUIRED_SLOTS = ("problem", "target", "goal")
OPTIONAL_SLOTS = tuple(s for s in ALL_SLOTS if s not in REQUIRED_SLOTS)
```
> **필수 ≠ 먼저 질문.** `goal`은 필수(게이트 조건)지만 질문은 7번째 — 솔루션·수익모델을 모르면 측정 가능한 목표가 안 나오므로 일부러 늦췄다. "필수냐"(게이트 멤버십)와 "몇 번째로 묻느냐"(질문 순서)를 분리.
> `advantage`(차별점·경쟁우위)는 기획서 9슬롯 외에 도메인 보강으로 추가한 슬롯 — "왜 우리인가".

**슬롯 정의의 단일 원천 = `SLOT_SPECS`** (`state.py`). 각 슬롯에 `{title, definition, boundary, question}`을 두고, `slot_guide_text()`가 이를 `"- solution (솔루션): <정의> | 경계: <경계규칙>"` 한 블록으로 렌더한다. **segment·extract_slot_fills·correction 프롬프트가 모두 이 한 함수를 임베드**해 같은 정의·경계를 공유한다(예전엔 정의가 네 곳에 흩어져 같은 내용이 호출마다 다른 슬롯에 들어가곤 했다). `boundary`("이건 여기 NOT 저기")는 헷갈리는 이웃 슬롯 경계를 못박은 것 — `solution↔revenue`(무엇을 만드나 vs 어떻게 버나), `goal↔revenue`(목표 수치 vs 과금 방식), `market↔advantage`(경쟁사 데이터 vs 우리 우위) 등 8쌍. 이 경계는 동시에 fill이 "한 슬롯에 깔끔히 안 떨어지면 `ambiguous`로 보류"하는 판정 근거가 된다(§3.4).

각 `Slot` = `{value, source_label, status}`.
- `source_label` ∈ `SourceLabel` (`common/schema/labels.py`): `user / research / empty` (3종)
  → **출처 라벨은 기획서 3.3 그대로 유지** (에이전트 이름은 logic_validator로 바뀌었지만 라벨 enum은 불변).
- `status` ∈ `empty / needs_clarification / filled`.

### 발화 유형 6종 (`UtteranceType`)

`clarification_needed · claim · opinion · correction · question · meta`
> `claim`은 사실·가설·결정·제약을 한 유형으로 묶는다 — 라우팅이 동일(research+rag+logic_validator)하기 때문. 주장을 어떻게 분해·검증할지는 **오케가 아니라 리서치 쿼리 분해기**가 정한다.
> 세그먼트마다 `in_scope`(계획 관련 여부)도 함께 매겨, false면 워커를 막고 부드럽게 리다이렉트.

### 라우트 (`Route`)

`research · rag · logic_validator · clarify · none`
> v0.7.5에서 `inference` → `critic` → **`logic_validator`** 로 변경. claim ↔ 사내 근거(RAG)의 논리적 지지 여부를 판정한다(validator 엔진).

### Segment / ValidationReport / Correction

- `Segment`: `{text, canonical_text, utterance_types[], in_scope, target_slot, routes[]}`
  - 처리 순서·그래프 분기는 **별도 priority 필드 없이** `routes`/`utterance_types`에서 직접 파생한다 — 워커 호출·디스패치 분기는 `routes`(워커 라우트 유무), 정정 처리는 `utterance_types`(`"correction"` 포함 여부).
  - `in_scope`: 사업 계획과 관련 있는 발화인가. False면 classify가 `routes=["none"]`로 막고 conversation이 `redirect` intent로 부드럽게 넘긴다. 기본 True.
  - 주장을 어떻게 분해·검증할지(검증 강도, 전제 vs 결론)는 **오케가 정하지 않는다** — 리서치 클러스터의 쿼리 분해기 몫.
- `ValidationReport`: `{subject, findings[], sources[], agreement, cluster}`, `cluster ∈ research/rag/logic_validator`.
- `Correction`: `{slot, previous, new, turn}` — 정정 이력.

### PlanState

그래프가 노드 사이로 주고받는 한 턴의 모든 것: `session_id, turn, user_input, messages[],
turn_segments[], slots{}, correction_log[], validation_reports[], turn_validation_reports[],
pending_clarifications[], pending_question, output_request, pending_confirmations[]`.
`initial_state()`가 빈 한 벌을 만든다 (슬롯 10개 모두 empty).
> `pending_confirmations[]`는 **애매해서 주입을 보류한 슬롯 값 큐**(`PendingConfirmation`). fill이
> `ambiguous`로 본 값을 슬롯 대신 여기 쌓고, 다음 턴 `confirm_resolve`가 사용자 답으로 해소한다.
> 다른 턴 임시필드와 달리 `run_turn`이 리셋하지 않아 **턴을 넘어 영속**(세션 스토어가 통째 저장).
> `validation_reports`는 **누적**, `turn_validation_reports`는 **이번 턴 dispatch 결과만**(매 턴 리셋).
> 대화 에이전트의 결과 보고가 '방금 돌린 것'만 보도록 분리(`turn_validation_reports`).
> SSE 에이전트 활동(`agent_start`→`validation_report`)은 `_stream`이 사후 재생하지 않고
> **dispatch가 워커 호출 직전/직후에 실시간 emit**한다 — `progress.py`의 ContextVar emitter가
> `chat.py`의 `asyncio.Queue`로 들어가고, `_stream`이 `run_turn`과 동시에 큐를 비워 흘린다.

---

## 2. 그래프 토폴로지 (`graph.py`)

```
START
 └▶ confirm_resolve 보류된 슬롯 확인 해소 (pending 있을 때만; 없으면 no-op) (LLM)
 └▶ segment        발화를 의미 단위로 분해 + 맥락 복원 (LLM)
 └▶ classify       각 세그먼트 다중 라벨 + 라우팅 결정 (LLM + 결정론 덮어쓰기)
 └▶ correction     정정 신호 처리 → 슬롯 clear/replace (LLM)
 └▶ [clarify_branch]  ← 유일한 조건부 엣지
      ├─ dispatch  research·rag 병렬 → logic_validator 후속 (2단계, 워커 라우트 있을 때)
      │    └▶ extract_fills   빈 슬롯에 값 추출 (LLM)
      │         └▶ gate
      └─ gate      (명확화만 있으면 dispatch 우회)
 └▶ gate           출력 의도 판정 + Type 0/1/2 분기 (LLM)
 └▶ conversation   state→intent 목록(_build_intents) 후 한 응답으로 렌더 (대화 에이전트, LLM)
 └▶ integrator     pending_clarifications만 기록 (pass-through, 결정론)
 └▶ END
```

- `build_graph()`는 `@lru_cache(maxsize=1)` — 한 번만 컴파일, 모든 턴이 공유.
- **`_clarify_branch`**: 명확화(`clarify` 라우트)만 있고 부를 워커(research/rag/logic_validator 라우트)가 하나도 없으면 `gate`로 직행 → 모호한 발화를 검증하지 않음 (기획서 6장 ②순위 규칙). `dispatch_node`와 같은 '워커 라우트 유무' 기준이라 부를 워커가 있으면 명확화가 섞여 있어도 dispatch로 보낸다.
- **`run_turn(state, user_input)`**: 진입점. 턴 카운터 증가 → user 메시지 적재 → 턴 임시필드 초기화 → `graph.ainvoke` → assistant 응답을 messages에 누적.

---

## 3. 노드별 명세

### 3.0 `confirm_resolve_node` (`nodes/confirm.py`)

**토폴로지상 맨 앞(START 직후)** — `pending_confirmations`가 비어 있으면 no-op이라 일반 턴엔 영향이 없다. 큐에 보류 건이 있으면(직전 턴에 fill이 애매하다고 판단해 쌓아둔 것), 이번 사용자 발화를 그 확인 질문에 대한 답으로 보고 LLM이 판정:
- `pick`: 후보 슬롯 중 하나를 고르거나 긍정 → 그 슬롯에 보류값 주입(이미 찬 슬롯이면 덮지 않음) + 큐에서 제거.
- `reject`: "아니/빼" 등 부정 → 큐에서 제거(슬롯은 빈 채).
- `unclear`: 그 질문과 무관한 다른 얘기 → `attempts++`; 한도(2회) 넘으면 제안 슬롯으로 자동 확정(무한 재질문 방지), 아니면 유지(다음 턴 재질문).

해소 후에도 파이프라인은 계속 흐른다 — 같은 발화에 추가 정보가 있으면 segment 이하가 정상 처리하고, 방금 채운 슬롯은 더 이상 empty가 아니라 fill이 다시 건드리지 않는다.

### 3.1 `segment_node` (`nodes/segment.py`)

긴 발화를 **의미 단위로 분해**하고 각 조각을 **자기충족 문장(`canonical_text`)** 으로 복원.
- 입력 프롬프트: 현재 슬롯 스냅샷 + 최근 대화 10턴 + 이번 발화.
- **맥락 복원 강화**: 다턴에 걸친 지시어 추적(직전 턴이 아니어도 선행사 연결, 정정 후 값 우선), 확정 고유명사·수치 우선 복원("거기 시장"→"일본 시장"), 미슬롯 주체 복원. 단서 없으면 원문 유지(환각 금지).
- LLM이 `{text, canonical_text, target_slot_hint}` 배열 반환.
- 프롬프트에 `slot_guide_text()`(정의·경계)를 임베드 — `target_slot_hint`를 슬롯 *이름*이 아니라 *정의*로 고른다. 10개 화이트리스트 검증, 경계가 헷갈리면 null로 두고 슬롯 확정은 fill에 위임.
- 빈 발화면 `[]`, 세그먼트 0개면 원문 1개로 폴백.

### 3.2 `classify_node` (`nodes/classify.py`)

각 세그먼트에 **다중 라벨**(`utterance_types`) 부여 후, **결정론 매트릭스**로 라우팅 확정.
> LLM이 매트릭스를 어기면 코드 룰(`derive_routes`)이 이긴다 — 모델 출력은 못 믿어도 비즈니스 룰은 코드로 박는다.

**라우팅 매트릭스 (`_ROUTE_MATRIX`, 기획서 5장):**

| 발화 유형 | clarify | research | rag | logic_validator |
|---|:---:|:---:|:---:|:---:|
| clarification_needed | ● | | | |
| claim | | ● | ● | ● |
| opinion | | | ● | ● |
| question | | ● | ● | |
| correction | (correction_node 처리) | | | |
| meta | (워커 호출 없음) | | | |

> `claim`(사실·가설·결정·제약)은 research+rag+logic_validator 모두 발동 — 라우팅이 같아 한 유형으로 묶는다.
> 검증 세부(전제 vs 사실 vs 결정 배경)는 오케가 아니라 리서치 쿼리 분해기가 claim·slot_context를 보고 정한다.

- `derive_routes`: 여러 라벨의 활성 클러스터 **합집합**, `[clarify, research, rag, logic_validator, none]` 순서로 정렬.
- segment가 미리 박은 라벨 보존 + LLM 추가 라벨 머지(화이트리스트·중복 제거), 둘 다 없으면 `opinion` 기본값.
- LLM 호출은 **세그먼트 전체 배치 1회** (호출 절약), 개수 어긋나면 LLM 결과 폐기.

### 3.3 `correction_node` (`nodes/correction.py`)

`utterance_types`에 `correction` 있는 세그먼트만 모아 LLM이 **슬롯 갱신 액션** 결정.
- `clear`: 슬롯 비움 + correction_log 적재.
- `replace`: `new_value`로 교체 + `source_label=USER` + log 적재.
- `ignore`: 모호하면 패스.
- 슬롯명은 10개 화이트리스트 검증. 타겟 없으면 첫 액션 슬롯을 세그먼트에 표시.
- `_CORR_SYSTEM`도 `slot_guide_text()`를 임베드 — 정정 시 슬롯 매칭이 fill·segment와 같은 정의·경계를 쓴다.

### 3.4 `extract_slot_fills_node` (`nodes/correction.py`)

dispatch 경로에서만 실행 (그래프상 dispatch 다음). **비어있는 슬롯**에 들어갈 값을 세그먼트에서 추출하고, **슬롯 선택의 단일 권위**다(segment의 `target_slot` 힌트는 payload에 prior로만 넘겨 두 판단이 갈리는 걸 줄인다).
- 후보 = `claim/opinion` 라벨 가진 세그먼트. 빈 슬롯 없으면 LLM 호출 안 함 (비용 절약).
- `_FILL_SYSTEM`에 `slot_guide_text()` 임베드. LLM은 fill마다 `{slot, value, confidence(clear|ambiguous), alt_slots[], reason}` 반환.
- **명확(`clear`)** → 빈 슬롯에 즉시 주입(`source_label=USER`). 이미 찬 슬롯은 correction_node 담당.
- **애매(`ambiguous` 또는 `alt_slots` 있음)** → 주입하지 않고 `pending_confirmations`에 한 건 쌓음(후보 중 빈 슬롯이 하나도 없으면 스킵). 다음 턴 `confirm_resolve`(§3.0)가 사용자 답으로 확정. → "같은 내용이 다른 슬롯에 들어가는" 문제를 (a)경계 명문화 (b)선택 단일화 (c)애매 시 사용자 확인으로 막는다.

### 3.5 `parallel_dispatch_workers_node` (`nodes/dispatch.py`)

**워커 라우트(`research/rag/logic_validator`)를 가진** 세그먼트를 보고 워커를 **2단계로 호출**.
> 라우트 유무로 판단 — `opinion`(routes=`rag·logic_validator`)도 매트릭스대로 디스패치된다.
> 명확화-only 턴은 `_clarify_branch`가 dispatch 자체를 우회하므로 보류된다.

**2단계 디스패치** (리서치·RAG 병렬 → 논리검증 후속):
```python
# 1단계: 외부 사실 + 회사 문서 — 전 세그먼트 병렬
research → run_research(subject) → report              # asyncio.gather; report를 idx로 보관
rag      → run_rag_check(subject) → (report, rag_result)
# 2단계: 논리검증 — 같은 세그먼트의 1단계 RAG 산출물 + (claim이면) research report를 입력으로
logic_validator → run_logic_validator(subject, rag_result, research_report)   # asyncio.gather
```
> **왜 2단계인가** — `logic_validator`(validator 엔진)는 RAG가 회수한 근거(highlight·raw_source)가
> claim을 논리적으로 지지하는지 판정하므로, 1단계 RAG 산출물 `RagExtractorResult`가 먼저 있어야
> 한다. RAG는 `(ValidationReport, RagExtractorResult)`를 돌려주고 dispatch가 그 원본을 2단계로
> 넘긴다(프론트엔 ValidationReport만 발행 — raw_source 등 대용량 제외). 매트릭스상 logic_validator는
> 항상 rag와 동반하므로 입력 근거는 늘 존재하며, RAG가 근거를 못 찾으면(`rag_result=None`) "근거
> 없음"으로 흐른다.
>
> **research 보조 근거 통합**: claim 세그먼트에선 같은 세그먼트의 1단계 research 결과(findings·sources)를
> 텍스트로 묶어 보조 근거로 함께 넘긴다. validator는 `research_evidence`(Optional[str]) 인자로 하위호환
> 확장돼 — 인자 없으면 user_msg가 기존과 바이트 동일 — **사내 RAG 근거를 1차, 외부 research를 보조**로
> 본다. opinion(research 라우트 없음)은 `research_report=None`으로 폴백해 기존대로 사내 근거만으로 판정.
>
> **역할 분담**: `rag`는 retrieval만(`agreement=unknown`), `logic_validator`는 판정만
> (verdict→agreement: supports→confirms / contradicts→contradicts / insufficient→partial /
> unrelated→unknown). 결과는 `turn_validation_reports`에 적재.
>
> **워커 구현 상태(실 구현)**: 세 워커 모두 실 파이프라인이다 — `research`(분해→검색→리포트),
> `rag`(rag_extractor: claim추출→폴더라우팅→Chroma 검색+하이라이트), `logic_validator`(validator
> 엔진 `run_validator`). 동기·블로킹 호출은 `asyncio.to_thread`로 감싼다. **mock 경로는 없다** —
> `OPENAI_API_KEY`가 없으면 실행 자체가 막힌다(§4).

### 3.6 `gate_node` (`nodes/gate.py`)

출력 게이트 — 기획서 8장 Type 0/1/2.
1. `detect_output_request`: LLM 1회로 `wants_output` 판정 (키워드 false positive 회피).
2. false면 `output_request=None` (일반 대화 턴).
3. 분기 (결정론):

| 조건 | 결과 |
|---|---|
| 필수 슬롯(P·T·G) 미달 | **Type 0** — 출력 거절 |
| 선택 슬롯도 다 참 | **Type 1** — 정상 완료 |
| 그 외 (필수만 참) | **Type 2** — 조기 출력 (빈칸 `[미정]`) |

`required_missing` / `optional_missing` 헬퍼는 다른 노드도 재사용.
- 출력 요청(`wants_output`)이 잡히면 `pending_confirmations`를 비운다 — 사용자가 진행을 택했으니 보류 중인 슬롯 확인은 흘려보낸다(출력 흐름과 충돌 방지).

### 3.7 `conversation_node` (`agents/conversation/agent.py`)

state에서 **intent 목록을 결정론으로 뽑아**(`_build_intents`) **LLM 1회로 한 응답으로 렌더** (대화 에이전트). conversation_spec TRIGGER MATRIX 전체를 지원:
`ask_slot · confirm_slot · clarify · report_findings · answer_question · redirect · reject_output · acknowledge · deliver_plan`.
> 구현 차이: conversation_spec은 `report_research`·`report_critique`를 별도 intent로 두지만, 코드는 한 주제의 research·rag·logic_validator 결과를 **`report_findings` 하나로 통합**해 넘긴다(렌더 프롬프트가 출처별로 구분). spec이 "둘은 한 턴에 묶일 수 있다(통합은 integrator 몫)"고 한 것을 그대로 반영.
- **intent 선택(결정론)**: 이번 턴 정정→`acknowledge`, `pending_confirmations`→`confirm_slot`(보류값과 후보 슬롯 제시), `in_scope=false`→`redirect`, `turn_validation_reports`→주제별 `report_findings`(claim·opinion) 또는 `answer_question`(question), `output_request`→`reject_output`(type0)·`deliver_plan`(type1/2), `clarify` 라우트→`clarify`. 위에서 막지 않았고 **확인 대기(`confirm_slot`)도 없으면** `ALL_SLOTS` 첫 빈칸으로 `ask_slot`(confirm_slot·clarify·type0·deliver가 있으면 다음 질문 보류).
- **렌더(LLM)**: intent 목록 JSON을 받아 한 메시지로 매끄럽게 연결(예: 결과 보고 → 다음 질문). 슬롯별 질문 톤은 `SLOT_SPECS[...]["question"]`(단일 원천)에서 가져와 `ask_slot.example`로 주입.
- **분류·판단은 안 함** — 무엇을 보고/질문할지는 state에서 파생, 대화는 표현만.

### 3.8 `response_integrator_node` (`nodes/integrator.py`)

**결정론 pass-through**. 응답을 만드는 일은 이제 `conversation_node`가 intent 목록으로 끝내므로,
통합기는 `pending_question`을 다시 만들지 않는다(대화 에이전트 결과를 덮어쓰지 않음).
세션 표시·디버깅용 `pending_clarifications`(이번 턴 `clarify` 세그먼트 목록)만 추려 기록.

---

## 4. LLM 호출 헬퍼 (`llm.py`)

모든 LLM 노드는 `call_json(system, user, schema)` 하나만 부른다. **mock/live 모드 분기는 없다** —
`OPENAI_API_KEY`가 없으면 `call_json`이 즉시 `RuntimeError`를 던지고 `api_server`도 기동 시점에
거부한다(fail-fast). 키가 있으면 항상 실 호출.
- `langchain-openai ChatOpenAI`, JSON 모드 + 스키마 힌트 주입 + pydantic 검증, 실패 시 1회 재시도.
- 3단 방어: 프롬프트에 스키마 박기 → JSON 모드 → pydantic `model_validate`.
- 모델 교체: `BPM_LLM_MODEL`(오케스트레이터), `OPENAI_MODEL`(리서치/RAG).

---

## 5. 매 턴 처리 패턴 (메시지 종류별)

| 메시지 종류 | 워커 호출 | 슬롯 변경 | 분기 |
|---|---|---|---|
| 신규 단일 발화 | 라벨에 따라 | 잠재적 | 6유형 라벨링 → 매트릭스 |
| 신규 다중 발화 | 세그먼트별 병렬 | 잠재적 | 라우트별 분기 |
| 정정 신호 | (재검증 보류 — 아래 갭) | **필수** | correction_node 먼저 |
| 출력 요청 | Planner (게이트 통과 시) | 없음 | Type 0/1/2 |
| 스코프 밖 발화 | 없음 (리다이렉트) | 없음 | `in_scope=false` → routes none |
| 메타·단순응답 | 없음 | 없음 | "응"·"다음" 등 |
| 애매한 슬롯 값 | 라벨에 따라 | 보류→확인 후 | fill `ambiguous` → `confirm_slot` → 다음 턴 `confirm_resolve` |

> **신호 키워드 정확도**("말고"·"빼자"·"뽑아줘")가 성능의 큰 부분. 첫 단계인
> 메시지 종류 판단이 어긋나면 그 턴 전체가 어긋난다.

### 기획서 대비 보강·갭

- **보강 (코드 > 기획서)**: 발화 유형 `question` 추가(총 6종), 슬롯 `advantage`(차별점) 추가(총 10개).
- **알려진 갭 (코드 < 기획서)**: 정정(correction) 시 교체된 슬롯 값의 **재검증 미동작**.
  기획서 5장은 리서치·RAG '재발동'을 요구하지만 현재는 슬롯 덮어쓰기만 함
  (`correction.py`의 TODO). 실 워커 연결 시 구현 예정.

---

## 6. 설계 원칙 (코드에 박힌 것)

1. **상시 진입점** — 조건부가 아니라 모든 메시지가 오케스트레이터를 거친다.
2. **판단/표현 분리** — 무엇을 물을지(오케) vs 어떻게 물을지(대화).
3. **분류 일원화** — 6유형 라벨링은 오케 단독. 논리검증·워커는 라벨링된 발화를 입력으로만 받음.
4. **LLM은 제안, 코드는 결정** — 라우팅·분기·Type 판정은 결정론 함수가 최종 확정.
5. **새로 추가된 것만 처리** — 매 턴 전체 재계산 X. 정정 이력은 별도 추적.
6. **입력은 분해, 출력은 절제** — 세그멘테이션 + 다중 라벨 + 라우팅 / 응답은 명확화 + 핵심 질문 1~2개.
7. **필수 슬롯이 게이트** — P·T·G 미달이면 어떤 출력 요청도 거절(방어 코드).

---

## 7. 파일 맵

```
agents/orchestrator/
├─ graph.py              토폴로지 조립 + run_turn (진입점)
├─ llm.py                call_json (키 필수, 구조화 출력)
├─ ORCHESTRATOR.md       (이 문서)
└─ nodes/
   ├─ confirm.py         애매한 슬롯 주입 확인 해소 (pending 큐, START 직후)
   ├─ segment.py         세그멘테이션 + 맥락 복원
   ├─ classify.py        다중 라벨 + 라우팅 매트릭스
   ├─ correction.py      정정 해소 + 슬롯 채움 (애매하면 pending 큐로 보류)
   ├─ dispatch.py        리서치·RAG 병렬 → 논리검증 2단계 호출
   ├─ gate.py            출력 게이트 Type 0/1/2
   ├─ integrator.py      응답 통합 (결정론)
   └─ router.py          (deprecated)

agents/conversation/agent.py   대화 에이전트 (intent 선택 + 한 응답 렌더)
agents/research/               리서치 실 파이프라인 (분해→검색→리포트)
agents/rag/                    RAG retrieval 워커 (rag_extractor + worker 어댑터)
agents/logic_validator/        논리검증 워커 (validator 엔진 호출 어댑터)
agents/validator/              validator 엔진 (run_validator: claim↔근거 판정)
common/schema/{state,labels}.py  슬롯·라벨·타입 정의
```
