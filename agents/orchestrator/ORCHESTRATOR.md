# 오케스트레이터 로직 총정리 (구현 명세)

> 기획서(`agent_plan_v7_5.html`, `orchestrator_spec.html`) v0.7.5 를 코드로 옮긴 결과.
> **기획서를 먼저 읽고 이 문서를 읽으면** 오케스트레이터가 실제로 어떻게 동작하는지
> 노드 단위로 파악할 수 있다. 코드 위치는 모두 `agents/orchestrator/` 기준.

---

## 0. 한 문장 요약

매 사용자 메시지마다 **단일 진입점**으로 들어와, LangGraph 상태머신이
`segment → classify → correction → (dispatch → extract_fills) → gate → conversation → integrator`
순서로 흐르며 **분류·세그멘테이션·라우팅·슬롯 추적·출력 게이트**를 수행한다.
판단(무엇을 할지)만 오케스트레이터가 하고, 표현(자연어)·실행(검색·추론)은 워커가 맡는다.

---

## 1. 핵심 데이터 모델 (`common/schema/state.py`)

### 슬롯 10개

| 구분 | 슬롯 | 의미 |
|---|---|---|
| **필수 3** | `problem` · `target` · `goal` | 출력 게이트 조건 (P·T·G) |
| **선택 7** | `solution` · `advantage` · `market` · `revenue` · `milestones` · `risks` · `resources` | 비어도 출력 가능 |

```python
REQUIRED_SLOTS = ("problem", "target", "goal")
OPTIONAL_SLOTS = ("solution", "advantage", "market", "revenue", "milestones", "risks", "resources")
```
> `advantage`(차별점·경쟁우위)는 기획서 9슬롯 외에 도메인 보강으로 추가한 슬롯 — "왜 우리인가".

각 `Slot` = `{value, source_label, status}`.
- `source_label` ∈ `SourceLabel` (`common/schema/labels.py`): `user / research / inference / candidate / empty`
  → **출처 라벨은 기획서 3.4 그대로 유지** (에이전트 이름은 critic으로 바뀌었지만 라벨 enum은 불변).
- `status` ∈ `empty / needs_clarification / filled`.

### 발화 유형 6종 (`UtteranceType`)

`clarification_needed · claim · opinion · correction · question · meta`
> v0.7.5에서 9→6 통합: 구 `fact_claim · hypothesis · decision · constraint`를 **`claim`** 하나로 합침.
> claim의 세부 구분은 별도 `ClaimType`(`fact · hypothesis_premise · decision_context · market_fill`)으로 분리 — 리서치 전달용.
> `question`(사용자 정보 요청 → 리서치·RAG)은 기획서 8유형 외 도메인 보강으로 추가.

### 라우트 (`Route`)

`research · rag · critic · clarify · none`
> v0.7.5에서 `inference` → **`critic`** 으로 변경. 비평(Critic)은 추론 점검 + 정합성 판단을 함께 맡는다.

### Segment / ValidationReport / Correction

- `Segment`: `{text, canonical_text, utterance_types[], claim_type, target_slot, routes[], priority}`
  - `priority`: **0=정정, 1=명확화, 2=워커 디스패치(claim/question), 3=의견·메타** — 그래프 분기 키.
  - `claim_type`: `claim` 유형일 때만 채워지는 세부 분류(`ClaimType`), 그 외 None.
- `ValidationReport`: `{subject, findings[], sources[], agreement, cluster}`, `cluster ∈ research/rag/critic`.
- `Correction`: `{slot, previous, new, turn}` — 정정 이력.

### PlanState

그래프가 노드 사이로 주고받는 한 턴의 모든 것: `session_id, turn, user_input, messages[],
turn_segments[], slots{}, correction_log[], validation_reports[], pending_clarifications[],
pending_question, output_request`.
`initial_state()`가 빈 한 벌을 만든다 (슬롯 10개 모두 empty).

---

## 2. 그래프 토폴로지 (`graph.py`)

```
START
 └▶ segment        발화를 의미 단위로 분해 + 맥락 복원 (LLM)
 └▶ classify       각 세그먼트 다중 라벨 + 라우팅 결정 (LLM + 결정론 덮어쓰기)
 └▶ correction     정정 신호 처리 → 슬롯 clear/replace (LLM)
 └▶ [clarify_branch]  ← 유일한 조건부 엣지
      ├─ dispatch  research/rag/critic 워커 병렬 호출 (priority=2 있을 때)
      │    └▶ extract_fills   빈 슬롯에 값 추출 (LLM)
      │         └▶ gate
      └─ gate      (명확화만 있으면 dispatch 우회)
 └▶ gate           출력/자동채움 의도 판정 + Type 0/1/2/3 분기 (LLM)
 └▶ conversation   다음 질문 한 문장 생성 (대화 에이전트, LLM)
 └▶ integrator     명확화 + 워커 통지 + 다음 질문 합치기 (결정론)
 └▶ END
```

- `build_graph()`는 `@lru_cache(maxsize=1)` — 한 번만 컴파일, 모든 턴이 공유.
- **`_clarify_branch`**: 명확화(priority 1)만 있고 디스패치 대상(priority 2)이 없으면 `gate`로 직행 → 모호한 발화를 검증하지 않음 (기획서 6장 ②순위 규칙).
- **`run_turn(state, user_input)`**: 진입점. 턴 카운터 증가 → user 메시지 적재 → 턴 임시필드 초기화 → `graph.ainvoke` → assistant 응답을 messages에 누적.

---

## 3. 노드별 명세

### 3.1 `segment_node` (`nodes/segment.py`)

긴 발화를 **의미 단위로 분해**하고 각 조각을 **자기충족 문장(`canonical_text`)** 으로 복원.
- 입력 프롬프트: 현재 슬롯 스냅샷 + 최근 대화 6턴 + 이번 발화.
- LLM이 `{text, canonical_text, target_slot_hint, hints[]}` 배열 반환.
- **`hints`로 선분류**: `correction`→priority 0, `clarification`→1, `meta`→3 을 미리 박아둠 (classify가 보존·보강).
- `target_slot_hint`는 10개 슬롯 화이트리스트로 검증, 아니면 null.
- 빈 발화면 `[]`, 세그먼트 0개면 원문 1개로 폴백.

### 3.2 `classify_node` (`nodes/classify.py`)

각 세그먼트에 **다중 라벨**(`utterance_types`) 부여 후, **결정론 매트릭스**로 라우팅 확정.
> LLM이 매트릭스를 어기면 코드 룰(`derive_routes`)이 이긴다 — 모델 출력은 못 믿어도 비즈니스 룰은 코드로 박는다.

**라우팅 매트릭스 (`_ROUTE_MATRIX`, 기획서 5장):**

| 발화 유형 | clarify | research | rag | critic |
|---|:---:|:---:|:---:|:---:|
| clarification_needed | ● | | | |
| claim | | ● | ● | ● |
| opinion | | | ● | ● |
| question | | ● | ● | |
| correction | (correction_node 처리) | | | |
| meta | (워커 호출 없음) | | | |

> 구 `fact_claim/hypothesis/decision/constraint`는 `claim`으로 통합 — research+rag+critic 모두 발동.
> claim의 검증 세부(전제 vs 사실 vs 결정 배경)는 `ClaimType`으로 리서치에 전달.

- `derive_routes`: 여러 라벨의 활성 클러스터 **합집합**, `[clarify, research, rag, critic, none]` 순서로 정렬.
- `derive_priority`: correction>clarification>dispatch>의견/메타 순.
- segment가 미리 박은 라벨 보존 + LLM 추가 라벨 머지(화이트리스트·중복 제거), 둘 다 없으면 `opinion` 기본값.
- LLM 호출은 **세그먼트 전체 배치 1회** (호출 절약), 개수 어긋나면 LLM 결과 폐기.

### 3.3 `correction_node` (`nodes/correction.py`)

`utterance_types`에 `correction` 있는 세그먼트만 모아 LLM이 **슬롯 갱신 액션** 결정.
- `clear`: 슬롯 비움 + correction_log 적재.
- `replace`: `new_value`로 교체 + `source_label=USER` + log 적재.
- `ignore`: 모호하면 패스.
- 슬롯명은 10개 화이트리스트 검증. 타겟 없으면 첫 액션 슬롯을 세그먼트에 표시.

### 3.4 `extract_slot_fills_node` (`nodes/correction.py`)

dispatch 경로에서만 실행 (그래프상 dispatch 다음). **비어있는 슬롯**에 들어갈 값을 세그먼트에서 추출.
- 후보 = `claim/opinion` 라벨 가진 세그먼트.
- 빈 슬롯 없으면 LLM 호출 안 함 (비용 절약).
- 추출값은 `source_label=USER`, 빈 슬롯에만 채움 (이미 찬 슬롯은 correction_node 담당).

### 3.5 `parallel_dispatch_workers_node` (`nodes/dispatch.py`)

**워커 라우트(`research/rag/critic`)를 가진** 세그먼트를 보고 워커 **병렬 호출** (`asyncio.gather`).
> priority가 아니라 라우트 유무로 판단 — `opinion`(priority 3, routes=`rag·critic`)도 매트릭스대로
> 디스패치된다. 명확화-only 턴은 `_clarify_branch`가 dispatch 자체를 우회하므로 보류된다.
```python
research → run_research(subject)
rag      → run_rag_check(subject)
critic   → run_critic(subject, slots)   # 비평만 슬롯 read-only 함께 받음
```
> v0.7.5: "검증" 단계가 사라지고 세 갈래 워커 디스패치로 분리. **비평(Critic)은 라벨링된
> 발화 + 슬롯 상태(read-only)** 를 받음 (기획서 11장 1주차 픽스 #3). 결과는
> `validation_reports`에 누적. 워커는 현재 모두 **stub** (실제 검색·추론은 다음 단계).

### 3.6 `gate_node` (`nodes/gate.py`)

출력 게이트 — 기획서 8장 Type 0/1/2/3.
1. `detect_output_request`: LLM 1회로 `wants_output` / `wants_autofill` 판정 (키워드 false positive 회피).
2. 둘 다 false면 `output_request=None` (일반 대화 턴).
3. 분기 (결정론):

| 조건 | 결과 |
|---|---|
| 필수 슬롯(P·T·G) 미달 | **Type 0** — 출력 거절 |
| 자동 채움 요청 | **Type 3** — 선택 슬롯 에이전트가 채움 |
| 선택 슬롯도 다 참 | **Type 1** — 정상 완료 |
| 그 외 (필수만 참) | **Type 2** — 조기 출력 (빈칸 `[미정]`) |

`required_missing` / `optional_missing` 헬퍼는 다른 노드도 재사용.

### 3.7 `conversation_node` (`agents/conversation/agent.py`)

오케가 정한 **"다음 채울 슬롯 + 모드"** 를 자연어 한 문장으로 변환 (대화 에이전트).
- 모드 결정: `type0`→거절, 필수 부족→`required`, 선택 부족→`optional`, 다 참→출력 권유(LLM 없이 고정 문구).
- 슬롯별 **few-shot 톤 예시**(`_FEW_SHOT`)를 프롬프트에 주입.
- **분류는 안 함** — 오케스트레이터가 끝낸 결정을 톤·길이만 입혀 표현.

### 3.8 `response_integrator_node` (`nodes/integrator.py`)

**LLM 호출 없는 결정론** 통합기. 한 문단으로 합침:
- 명확화(priority 1) 세그먼트 → "먼저 명확히 — …" (최대 2개).
- 디스패치(priority 2) 세그먼트 → "…쪽은 백그라운드에서 …" 통지.
- 명확화가 있으면 다음 질문은 보류(다음 턴), 없으면 워커 통지 + `conversation`의 질문 순.

---

## 4. LLM 호출 헬퍼 (`llm.py`)

모든 LLM 노드는 `call_json(system, user, schema)` 하나만 부른다.
- **mock 모드** (`BPM_LLM_MODE=mock` 또는 키 없음): `register_mock(key, handler)`로 등록된 가짜 응답. key = system 프롬프트 **첫 줄**.
- **live 모드**: `langchain-openai ChatOpenAI`, JSON 모드 + 스키마 힌트 주입 + pydantic 검증, 실패 시 1회 재시도.
- 3단 방어: 프롬프트에 스키마 박기 → JSON 모드 → pydantic `model_validate`.

---

## 5. 매 턴 처리 패턴 (메시지 종류별)

| 메시지 종류 | 워커 호출 | 슬롯 변경 | 분기 |
|---|---|---|---|
| 신규 단일 발화 | 라벨에 따라 | 잠재적 | 9유형 라벨링 → 매트릭스 |
| 신규 다중 발화 | 세그먼트별 병렬 | 잠재적 | 우선순위 적용 후 |
| 정정 신호 | (재검증 보류 — 아래 갭) | **필수** | correction_node 먼저 |
| 출력 요청 | Planner (게이트 통과 시) | 없음 | Type 0/1/2 |
| 자동 채움 요청 | 리서치 + 결정형 후보 | **필수** | Type 3 |
| 메타·단순응답 | 없음 | 없음 | "응"·"다음" 등 |

> **신호 키워드 정확도**("말고"·"빼자"·"뽑아줘"·"알아서")가 성능의 큰 부분. 첫 단계인
> 메시지 종류 판단이 어긋나면 그 턴 전체가 어긋난다.

### 기획서 대비 보강·갭

- **보강 (코드 > 기획서)**: 발화 유형 `question`(8→9), 슬롯 `advantage`(차별점, 9→10) 추가.
- **알려진 갭 (코드 < 기획서)**: 정정(correction) 시 교체된 슬롯 값의 **재검증 미동작**.
  기획서 5장은 리서치·RAG '재발동'을 요구하지만 현재는 슬롯 덮어쓰기만 함
  (`correction.py`의 TODO). 실 워커 연결 시 구현 예정.

---

## 6. 설계 원칙 (코드에 박힌 것)

1. **상시 진입점** — 조건부가 아니라 모든 메시지가 오케스트레이터를 거친다.
2. **판단/표현 분리** — 무엇을 물을지(오케) vs 어떻게 물을지(대화).
3. **분류 일원화** — 8유형 라벨링은 오케 단독. 비평·워커는 라벨링된 발화를 입력으로만 받음.
4. **LLM은 제안, 코드는 결정** — 라우팅·우선순위·Type 분기는 결정론 함수가 최종 확정.
5. **새로 추가된 것만 처리** — 매 턴 전체 재계산 X. 정정 이력은 별도 추적.
6. **입력은 분해, 출력은 절제** — 세그멘테이션 + 다중 라벨 + 우선순위 / 응답은 명확화 + 핵심 질문 1~2개.
7. **필수 슬롯이 게이트** — P·T·G 미달이면 어떤 출력 요청도 거절(방어 코드).

---

## 7. 파일 맵

```
agents/orchestrator/
├─ graph.py              토폴로지 조립 + run_turn (진입점)
├─ llm.py                call_json (mock/live, 구조화 출력)
├─ ORCHESTRATOR.md       (이 문서)
└─ nodes/
   ├─ segment.py         세그멘테이션 + 맥락 복원
   ├─ classify.py        다중 라벨 + 라우팅 매트릭스
   ├─ correction.py      정정 해소 + 슬롯 채움
   ├─ dispatch.py        리서치/RAG/비평 병렬 호출
   ├─ gate.py            출력 게이트 Type 0/1/2/3
   ├─ integrator.py      응답 통합 (결정론)
   └─ router.py          (deprecated)

agents/conversation/agent.py   대화 에이전트 (질문 생성)
agents/{research,rag,critic}/  워커 stub
common/schema/{state,labels}.py  슬롯·라벨·타입 정의
```
