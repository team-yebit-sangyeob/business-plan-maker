# 오케스트레이터 로직 총정리 (구현 명세)

> 기획서(`agent_plan_v7_5.html`, `orchestrator_spec.html`) v0.7.5 를 코드로 옮긴 결과.
> **기획서를 먼저 읽고 이 문서를 읽으면** 오케스트레이터가 실제로 어떻게 동작하는지
> 노드 단위로 파악할 수 있다. 코드 위치는 모두 `agents/orchestrator/` 기준.

---

## 0. 한 문장 요약

매 사용자 메시지마다 **단일 진입점**으로 들어와, LangGraph 상태머신이
`confirm_resolve → segment → classify → correction → (dispatch → extract_fills) → conversation → integrator`
순서로 흐르며 **(보류된 슬롯 확인 해소)·분류·세그멘테이션·라우팅·슬롯 추적**을 수행한다.
판단(무엇을 할지)만 오케스트레이터가 하고, 표현(자연어)·실행(검색·추론)은 워커가 맡는다.
계획서 생성은 이 그래프 밖이다 — 채팅은 슬롯을 채우고 답할 뿐, 계획서는 명시적 버튼(POST /plan)에서만 합성한다.

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

**슬롯 정의의 단일 원천 = `SLOT_SPECS`** (`state.py`). 각 슬롯에 `{title, definition, boundary, question}`을 두고, `slot_guide_text()`가 이를 `"- solution (솔루션): <정의> | 경계: <경계규칙>"` 한 블록으로 렌더한다. **extract_slot_fills·correction 프롬프트가 이 한 함수를 임베드**해 같은 정의·경계를 공유한다 — 슬롯 선택의 두 권위가 같은 문구를 본다(segment은 더는 슬롯을 고르지 않아 임베드하지 않는다, §3.1). `boundary`("이건 여기 NOT 저기")는 헷갈리는 이웃 슬롯 경계를 못박은 것 — `solution↔revenue`(무엇을 만드나 vs 어떻게 버나), `goal↔revenue`(목표 수치 vs 과금 방식), `market↔advantage`(경쟁사 데이터 vs 우리 우위) 등 8쌍. 이 경계는 동시에 fill이 "한 슬롯에 깔끔히 안 떨어지면 `ambiguous`로 보류"하는 판정 근거가 된다(§3.4). 도구·슬롯 메타질문(`tool_help`)의 응답도 같은 `SLOT_SPECS`에서 렌더한다 — `tool_help_text()`가 `APP_OVERVIEW`+슬롯 정의(제목·정의·경계) 전부를 한 덩이 '참고 자료'(body)로 묶고, conversation은 사용자 질문(subject)과 그 body만 받는다. **어느 슬롯을 얼마나 답할지(특정 1개/여럿/전체/개요)는 코드가 키워드로 가르지 않고 LLM이 질문에 맞춰 정한다** — 라우팅(워커 호출)이 아니라 표현 결정이라 LLM 몫이다. 슬롯 설명이 또 갈리지 않게 단일 원천을 공유한다.

각 `Slot` = `{value, source_label, status}`.
- `source_label` ∈ `SourceLabel` (`common/schema/labels.py`): `user / research / empty` (3종)
  → **출처 라벨은 기획서 3.3 그대로 유지** (에이전트 이름은 logic_validator로 바뀌었지만 라벨 enum은 불변).
- `status` ∈ `empty / needs_clarification / filled`.

### 발화 유형 8종 (`UtteranceType`) — content / interaction 2-tier

**content** (매트릭스로 워커 라우트 파생): `clarification_needed · claim · correction · question`
**interaction** (디스패치 없음 → conversation이 직접 처리, `routes=["none"]`): `meta · recall · tool_help · reason`
> classify가 세그먼트를 **먼저 interaction인지 content인지** 가른다 — 되묻기·진행신호처럼 말의 행위가 핵심인 발화가, 안에 낀 검증가능한 명제 때문에 claim으로 끌려가 워커가 발동하던 문제를 막는다. interaction은 "새 사실을 검색·검증할 게 없는, 대화 자체로 받는 발화"고, content는 "검증·명확화·정정·정보탐색이 필요한 발화"다.
> `claim`은 사실·가설·결정·제약·**근거 있는 가치판단**을 한 유형으로 묶는다 — 라우팅이 동일(research+rag+logic_validator)하기 때문(구 `opinion` 흡수: 라우팅 차이가 research 발동뿐이라 분리 가치 없음). 주장을 어떻게 분해·검증할지는 **오케가 아니라 리서치 쿼리 분해기**가 정한다.
> `recall`(되묻기)은 [최근 대화]에 이미 나온 걸 다시 묻는 발화 — 워커 없이 conversation이 대화 이력에서 답한다. 예: "아까 일본 된다며?", "우리 타겟 뭐로 정했지?".
> `tool_help`는 도구·슬롯·사용법 자체를 묻는 메타질문("솔루션 슬롯이 뭐야?", "각 슬롯의 역할?", "넌 뭐 할 수 있어?") — 워커 없이 conversation이 `SLOT_SPECS`·`APP_OVERVIEW`에서 `explain_tool`로 답한다(슬롯 정의 전체를 재료로 주고 답변 범위는 LLM이 질문에 맞춘다 — 코드가 스코프를 정하지 않음). classify가 interaction 라벨을 content와 섞어 줘도 코드(interaction-precedence)가 content를 떨궈 워커를 막는다. "솔루션 슬롯이 뭐야(도구 설명)"가 `question`으로 끌려가 리서치를 돌리던 문제를 막는다.
> `reason`(추론·제안 요청)은 **이미 모은 근거·대화 내용을 재료로 결론을 내달라**는 발화 — 새 검색 없이 conversation이 `session_evidence`(턴을 넘어 누적된 research·rag·logic_validator findings)와 대화 이력에서 직접 추론해 `reason_over_context` intent로 답한다(§3.7). 두 결을 한 유형으로 묶는다:
>   - **① 종합·도출** — "여기서 문제점 추론해봐", "방금 분석에서 도출할 게 뭐야?", "이걸로 시사점 정리해줘". 가진 근거를 엮어 정리한다.
>   - **② 제안·결정 요청** — "네가 생각하는 문제는 뭐야?", "넌 어떻게 봐?", "추천해줘", "정해줘", "제안해봐". 어시스턴트가 직접 후보를 제안한다.
>
>   둘 다 '새 외부 사실을 묻는 게 아니라 가진 것으로 답하라'는 점이 같아 라우팅이 동일(워커 0)하다. 핵심은 **`question`과의 경계**다 — 대화에 없던 *새 외부 대상*의 사실을 물으면 `question`(리서치를 돌린다), 가진 맥락이나 *어시스턴트의 판단*을 청하면 `reason`. 겉보기엔 둘 다 묻는 말이어도 `"커피 시장 규모 어때?"`는 새 외부 사실이라 question이고 `"그래서 네 생각엔 우리 문제가 뭔데?"`는 어시스턴트 판단을 청하니 reason이다. 이 경계가 흐려지면, 사용자가 "네 의견을 달라"는데 엉뚱하게 웹 검색을 돌리는 회피가 생긴다 — ②를 명문화한 이유다(§3.1·§3.7에서 segment·conversation이 이 신호를 어떻게 보존·소비하는지 이어 설명).
> 세그먼트마다 `in_scope`(계획 관련 여부)도 함께 매겨, false면 워커를 막고 부드럽게 리다이렉트.

### 라우트 (`Route`)

`research · rag · logic_validator · clarify · none`
> v0.7.5에서 `inference` → `critic` → **`logic_validator`** 로 변경. claim ↔ 사내 근거(RAG)의 논리적 지지 여부를 판정한다(validator 엔진).

### Segment / ValidationReport / Correction

- `Segment`: `{text, canonical_text, utterance_types[], in_scope, target_slot, routes[]}`
  - 처리 순서·그래프 분기는 **별도 priority 필드 없이** `routes`/`utterance_types`에서 직접 파생한다 — 워커 호출·디스패치 분기는 `routes`(워커 라우트 유무), 정정 처리는 `utterance_types`(`"correction"` 포함 여부).
  - `in_scope`: 사업 계획과 관련 있는 발화인가. False면 classify가 `routes=["none"]`로 막고 conversation이 `redirect` intent로 부드럽게 넘긴다. 기본 True.
  - 주장을 어떻게 분해·검증할지(검증 강도, 전제 vs 결론)는 **오케가 정하지 않는다** — 리서치 클러스터의 쿼리 분해기 몫.
- `ValidationReport`: `{subject, findings[], sources[], agreement, cluster, citations[]}`, `cluster ∈ research/rag/logic_validator`. `citations`=제목·URL·인용문·페이지·관련도(·사내문서 `raw_source`)까지 보존한 구조화 출처(SSE로 그대로 발행).
- `Correction`: `{slot, previous, new, turn}` — 정정 이력.

### PlanState

그래프가 노드 사이로 주고받는 한 턴의 모든 것: `session_id, turn, user_input, messages[],
turn_segments[], slots{}, correction_log[], turn_validation_reports[], turn_evidence[],
session_evidence[], pending_clarifications[], pending_question, pending_confirmations[],
open_proposal, confirmation_consumed, last_asked_slot, evidence_mode`.
`initial_state()`가 빈 한 벌을 만든다 (슬롯 10개 모두 empty).
> `pending_confirmations[]`는 **주입을 보류한 슬롯 값 큐**(`PendingConfirmation`). fill이 슬롯 애매(`confirm_kind="slot"`)·
> 결정 미확정(탐색, `confirm_kind="commit"`)·찬 슬롯과 충돌(교체, `confirm_kind="replace"`)로 본 값을 슬롯 대신 여기 쌓고,
> 다음 턴 `confirm_resolve`가 사용자 답으로 해소한다. **두 번째 등록 출처는 `conversation_node`** — reason 제안 모드②가
> 슬롯 값을 제안하면 그 제안을 여기 commit/replace로 쌓는다(§3.7). 한 슬롯에 여러 안을 제시하면 `candidate_values[]`에
> [추천, *대안]을 싣고, `confirm_resolve`가 `chosen_index`로 고른다(단일 안이면 비움).
> 다른 턴 임시필드와 달리 `run_turn`이 리셋하지 않아 **턴을 넘어 영속**(세션 스토어가 통째 저장).
> `open_proposal`·`confirmation_consumed`는 **이번 턴 전용**(`run_turn`이 매 턴 박는다): `open_proposal`은 턴 시작 시점의
> 열린 제안 스냅샷(`pending_confirmations[0]`) — `confirm_resolve`가 라이브 큐를 pop해도 segment·classify가 "이번 발화가
> 무엇에 대한 답인가"를 보게 한다(`conversation_state_text`로 렌더). `confirmation_consumed`는 그 발화를 순수 확인 답으로
> 소비했는지 — True면 그래프가 segment 이하를 건너뛴다(§3.0, §2).
> `turn_validation_reports`는 **이번 턴 dispatch 결과만**(매 턴 리셋) — 대화 에이전트의 결과 보고가
> '방금 돌린 것'만 보도록 분리. `turn_evidence`도 이번 턴치(리셋)이되 슬롯 연결정보를 더한 `EvidenceRecord`다.
> `run_turn`이 끝에서 `turn_evidence`를 `session_evidence`로 합친다(중복 제거 + 슬롯 백필) — `session_evidence`는
> **세션 전체 누적**이라 턴을 넘어 영속하며, 계획서(planner)가 출처를 인용하는 원천이다.
> SSE 에이전트 활동(`agent_start`→`validation_report`)은 `_stream`이 사후 재생하지 않고
> **dispatch가 워커 호출 직전/직후에 실시간 emit**한다 — `progress.py`의 ContextVar emitter가
> `chat.py`의 `asyncio.Queue`로 들어가고, `_stream`이 `run_turn`과 동시에 큐를 비워 흘린다.
> 같은 경로로 **노드 단계(`stage`) 이벤트**도 흐른다 — `graph._staged` 래퍼가 노드 시작 직전
> `{type:"stage", node, label}`을 emit해(dispatch·integrator 제외), 워커 안 도는 턴도 진행 단계가 보인다.

---

## 2. 그래프 토폴로지 (`graph.py`)

```
START
 └▶ confirm_resolve 보류된 슬롯 확인 해소 (pending 있을 때만; 없으면 no-op) (LLM)
 └▶ [post_confirm_branch]  ← 조건부 엣지 ①
      ├─ conversation  (순수 확인 답 소비 → segment 이하 우회: 열린 제안에 대한 답을 새 리서치로 재처리 안 함)
      └─ segment       발화를 의미 단위로 분해 + 맥락 복원 (LLM)
           └▶ classify     각 세그먼트 다중 라벨 + 라우팅 결정 (LLM + 결정론 덮어쓰기)
           └▶ correction   정정 신호 처리 → 슬롯 clear/replace (LLM)
           └▶ [clarify_branch]  ← 조건부 엣지 ②
                ├─ dispatch  research·rag 병렬 → logic_validator 후속 (2단계, 워커 라우트 있을 때)
                │    └▶ extract_fills   빈 슬롯 채움 · 교체 확인 · 근거 슬롯 태그 (LLM)
                │         └▶ conversation
                └─ conversation  (명확화만 있으면 dispatch·fills 우회)
 └▶ conversation   state→intent 목록(_build_intents) 후 한 응답으로 렌더 (대화 에이전트, LLM)
 └▶ integrator     pending_clarifications만 기록 (pass-through, 결정론)
 └▶ END
```

- `build_graph()`는 `@lru_cache(maxsize=1)` — 한 번만 컴파일, 모든 턴이 공유.
- **노드 재시도**: LLM 노드(`confirm_resolve`·`segment`·`classify`·`correction`·`extract_fills`·`conversation`)엔 `add_node(..., retry_policy=RetryPolicy(max_attempts=3))`로 전이 오류(429·timeout·5xx) 재시도를 단다(기본 backoff 2.0·jitter). `dispatch`는 제외 — 재시도가 리서치·RAG·논리검증 워커를 재호출해 비멱등·SSE 카드 중복을 부른다. `integrator`도 제외(LLM 없는 통과).
- **`_post_confirm_branch`** (조건부 엣지 ①, confirm_resolve 뒤): `confirm_resolve`가 이번 발화를 **순수 확인 답**(accept/pick/reject + 추가 내용 없음)으로 소비하면(`confirmation_consumed=True`) `conversation`으로 직행해 segment·classify·dispatch를 건너뛴다 — 열린 제안에 대한 답("이대로 넣어")을 새 리서치 주장으로 재처리하는 회귀를 막는다. revise·unrelated·추가 내용 섞인 답은 `segment`로 흘려 정상 처리한다(상태 주입을 받은 classify가 잔여 확인절을 `meta`로 떨궈 재디스패치를 막고, 새 내용은 평소대로 분류).
- **`_clarify_branch`** (조건부 엣지 ②, correction 뒤): 명확화(`clarify` 라우트)만 있고 부를 워커(research/rag/logic_validator 라우트)가 하나도 없으면 `conversation`으로 직행 → dispatch·extract_fills를 우회하고 모호한 발화를 검증하지 않음 (기획서 6장 ②순위 규칙). `dispatch_node`와 같은 '워커 라우트 유무' 기준이라 부를 워커가 있으면 명확화가 섞여 있어도 dispatch로 보낸다.
- **`run_turn(state, user_input)`**: 진입점. 턴 카운터 증가 → user 메시지 적재 → 턴 임시필드 초기화(`open_proposal`에 턴 시작 `pending_confirmations[0]` 스냅샷, `confirmation_consumed=False`) → `graph.ainvoke` → assistant 응답을 messages에 누적.

---

## 3. 노드별 명세

### 3.0 `confirm_resolve_node` (`nodes/confirm.py`)

**토폴로지상 맨 앞(START 직후)** — `pending_confirmations`가 비어 있으면 no-op이라 일반 턴엔 영향이 없다. 큐에 보류 건이 있으면, 이번 사용자 발화를 그 제안에 대한 답으로 보고 **태도(decision)**를 LLM이 판정한다(명령형 "이대로 넣어/넣어"도 제안에 대한 긍정이면 accept — 새 리서치 주문이 아니다; 보기는 예시지 정답 키워드가 아니라 의미로 판단):
- `accept`: 제안대로 넣으라는 긍정·명령 → 제안 슬롯에 주입(`replace`면 기존 값 덮어쓰기 + correction_log 기록; 그 외엔 이미 찬 슬롯을 덮지 않음) + 큐 제거. 다중 후보 값(`candidate_values`)이면 특정 안을 콕 집지 않은 수락이라 추천안(`value`=1순위)으로 채운다.
- `pick`: 제시한 후보 중 하나 선택 → 그 슬롯·그 값 주입 + 큐 제거. (a) 후보 **슬롯**이 둘 이상이면 `slot`으로 어느 칸인지, (b) 한 슬롯에 후보 **값**이 여럿(`candidate_values`, reason 제안 모드②)이면 `chosen_index`(1-base)로 어느 안인지 고른다. 제시한 안 어디에도 없는 새 값을 말하면 accept/pick이 아니라 `revise`로 본다 — 명령형 "~로 하자"여도 억지로 가까운 안에 매칭해 잘못된 값을 박지 않는다.
- `reject`: "아니/빼" 부정 → 큐에서 제거(슬롯 그대로).
- `revise`: 값을 고쳐서 넣으라 → 스테일 값 버리고 큐 제거, 주입 안 함(고친 값은 파이프라인이 다시 뽑음).
- `unrelated`: 그 질문과 무관한 다른 얘기 → `confirm_kind`로 가른다. `commit`(결정 미확정)·`replace`(교체 확인)는 드롭(결정/교체 안 한 건 안 건드린다), `slot`(값은 결정, 칸만 모름)은 `attempts++` 후 한도(2회) 넘으면 제안 슬롯으로 자동 확정(무한 재질문 방지).

**소비(consumed) 신호**: 순수 확인 답(accept/pick/reject + `has_additional_content=false`)이면 `confirmation_consumed=True`를 내보내 `_post_confirm_branch`가 segment 이하를 건너뛴다(§2) — 열린 제안에 대한 답을 새 리서치 주장으로 재처리하던 회귀를 끊는다. 추가 내용이 섞였거나(`has_additional_content`) revise·unrelated면 파이프라인이 계속 흐르고, 같은 발화의 새 내용은 segment 이하가 정상 처리한다(방금 채운 슬롯은 더 이상 empty가 아니라 fill이 다시 건드리지 않는다).
> 예전 `pick/reject/unclear` 3분류에선 명령형 "이대로 넣어"가 `unclear`로 떨어져 commit-kind 값을 **드롭**(값 손실)하고, 그 발화가 segment·dispatch로 흘러 다시 리서치됐다 — 이 두 버그를 태도 5분류 + consumed 분기로 함께 막는다.

### 3.1 `segment_node` (`nodes/segment.py`)

긴 발화를 **의미 단위로 나누고** 각 조각을 **그 문장만 봐도 뜻이 통하는 문장(`canonical_text`)** 으로 복원. **딱 두 가지만 한다 — 분할과 맥락 복원. 슬롯은 고르지 않는다**(예전엔 `target_slot_hint`까지 떠안아 `slot_guide_text()` 전체를 프롬프트에 싣느라, 모델이 슬롯·값에 과몰입해 확인 답("이대로 넣어")을 대기 값으로 부풀려 선언문으로 만들곤 했다 — 그게 그 발화를 claim으로 끌고 가 재리서치되는 회귀의 발화원이었다).
- 입력 프롬프트: 현재 슬롯 스냅샷 + **`[대화 상태]`**(`conversation_state_text`: 열린 제안·직전 질문 슬롯, 있을 때만) + 최근 대화 10턴 + 이번 발화. `slot_guide_text()`는 더 이상 싣지 않는다.
- **맥락 복원 강화**: 여러 턴에 걸친 지시어 추적(직전 턴이 아니어도 가리키는 대상 연결, 정정 후 값 우선), 확정된 고유명사·수치 우선 복원("거기 시장"→"일본 시장"), 슬롯이 비어 있어도 주어 복원(복원 대상은 사업 도메인 한정 — 대화 화자 '어시스턴트/사용자'는 주어로 넣지 않음). 단서가 없으면 원문 유지(없는 맥락은 지어내지 않음). 되묻기·확인 발화("아까 ~라 했잖아?")는 말한 주체를 지어내지 말고 되묻는 내용만 복원.
- **열린 제안에 대한 답**(`[대화 상태]`에 확인 대기·직전 질문 슬롯이 있을 때): "이대로 넣어/빼/솔루션에 넣어/고쳐서"는 사업 주장으로 부풀리지 말고(대기 값을 주어로 끌어오지 않음) 답의 뜻만 간결히 남긴다. 직전 질문 슬롯에 대한 짧은 답은 그 슬롯의 답으로 자연스럽게 복원.
- **어시스턴트에게 청하는 제안 요청 프레이밍 보존**: "네 생각엔 뭐가 좋아?"·"네가 생각하는 문제는 뭐야?"·"추천해줘"·"정해줘"처럼 어시스턴트의 판단을 청하는 발화는 일반 사실 질문이나 사업 주장으로 고쳐쓰지 않고 **'청하는' 틀을 유지**한다 — 무엇에 대한 제안인지(주제·슬롯)만 맥락으로 복원하고 "네 생각/추천" 지시는 그대로 둔다. 예(이전 맥락: 커피 사업): `"네 생각엔 우리 문제가 뭐야?"` → `"네 생각엔 커피 사업의 문제가 뭐야?"`(O) / `"커피 사업의 문제는 무엇인가?"`(X). **왜 중요한가** — segment→classify는 한 방향이라, segment가 이 신호를 지워 평범한 질문으로 만들면 classify가 `reason`이 아니라 `question`으로 보고 리서치를 돌린다(복구 불가). 실제로 이 규칙이 없을 때 "네 생각엔 우리 문제가 뭐야?"가 "커피 사업의 문제는 무엇인가?"로 정규화돼 워커가 잘못 발동하는 회귀가 통합 테스트(`test_no_worker_routing_live`)에서 잡혔다.
- LLM이 `{text, canonical_text}` 배열 반환. **`target_slot`은 항상 `None`으로 둔다** — 슬롯 확정은 fill·correction이 정의·경계로 한다(§3.4).
- 빈 발화면 `[]`, 세그먼트 0개면 원문 1개로 폴백.

### 3.2 `classify_node` (`nodes/classify.py`)

각 세그먼트에 **다중 라벨**(`utterance_types`) 부여 후, **결정론 매트릭스**로 라우팅 확정.
> LLM이 매트릭스를 어기면 코드 룰(`derive_routes`)이 이긴다 — 모델 출력은 못 믿어도 비즈니스 룰은 코드로 박는다.

**라우팅 매트릭스 (`_ROUTE_MATRIX`, 기획서 5장):**

| 발화 유형 | clarify | research | rag | logic_validator |
|---|:---:|:---:|:---:|:---:|
| clarification_needed | ● | | | |
| claim | | ● | ● | ● |
| question | | ● | ● | |
| correction | (correction_node 처리) | | | |
| meta | (interaction — 워커 호출 없음) | | | |
| recall | (interaction — conversation이 이력에서 답) | | | |
| tool_help | (interaction — conversation이 SLOT_SPECS·APP_OVERVIEW에서 답) | | | |
| reason | (interaction — conversation이 누적 근거·대화이력에서 직접 추론·제안) | | | |

> `claim`(사실·가설·결정·제약·근거 있는 가치판단)은 research+rag+logic_validator 모두 발동 — 라우팅이 같아 한 유형으로 묶는다(구 `opinion` 포함).
> interaction(`meta`·`recall`·`tool_help`·`reason`)은 빈 집합 → `derive_routes`가 `["none"]`. 키를 지우지 않고 빈 집합으로 두는 이유: 라벨이 `_VALID_TYPES` 필터를 통과해 살아남아야 conversation·correction이 그 라벨을 본다.
> **interaction-precedence**: interaction 라벨이 content 라벨과 한 세그먼트에 섞이면 `classify_node`가 content를 떨군다(`_INTERACTION_TYPES = {meta, recall, tool_help, reason}`). `derive_routes`가 라우트를 **합집합**해서 `["tool_help","claim"]`이면 claim의 워커 라우트가 살아 새기 때문 — 프롬프트도 이들을 단독으로 두라 하지만 "누구를 부를지는 코드"라 backstop으로 강제한다.
> 검증 세부(전제 vs 사실 vs 결정 배경)는 오케가 아니라 리서치 쿼리 분해기가 claim·slot_context를 보고 정한다.

- `derive_routes`: 여러 라벨의 활성 클러스터 **합집합**, `[clarify, research, rag, logic_validator, none]` 순서로 정렬.
- segment가 미리 박은 라벨 보존 + LLM 추가 라벨 머지(화이트리스트·중복 제거), 둘 다 없으면 `clarification_needed` 기본값(분류 실패 시 안전 폴백 — `_VALID_TYPES`를 통과하고 `clarify`만 타 워커 비용 0; claim 폴백은 실패 턴마다 워커를 터뜨린다).
- LLM 호출은 **세그먼트 전체 배치 1회**(호출 절약), payload에 **`[대화 상태]`**(`conversation_state_text`: 열린 제안·직전 질문 슬롯)·**`[현재 슬롯]`**(`slot_snapshot_text`: 충돌·되묻기 판정용 값)·최근 대화를 함께 실어 맥락에서 판단하게 한다(예전엔 `[최근 대화]`만 — 슬롯값을 못 봐 충돌을 인지 못 했다). 개수 어긋나면 LLM 결과 폐기.
- **상태 인지 규칙(프롬프트 가이드, 라우팅은 여전히 코드)**: ① 열린 제안에 대한 답(수락 "이대로 넣어"·거부 "빼"·슬롯 선택 "솔루션에 넣어")은 `meta`로 둔다 → 워커 미발동(딸린 새 내용은 별개 세그먼트로 평소대로). 순수 확인 답은 보통 `_post_confirm_branch`가 classify 전에 우회하지만, 추가 내용 섞인 fall-through 턴에서 잔여 확인절이 재디스패치되는 걸 이 규칙이 막는다. ② 직전 질문 슬롯에 대한 짧은 답은 `claim`(채울 답)으로 — 짧다고 `clarification_needed`로 떨구지 않는다. ③ 슬롯 기준에 비해 **공허한 답**(goal에 "결과물", target에 "사람들" — 수치·구체 대상 없음)은 `claim`이 아니라 `clarification_needed`로 라벨해 리서치 대신 되묻는다(충분성 게이트의 coarse 단계; 정밀 판정은 fill, §3.4). ④ **어시스턴트에게 판단·제안을 청하는 답**("네 생각엔 뭐가 좋아?", "추천해줘", "정해줘")은 새 외부 사실을 묻는 게 아니라 가진 것으로 답하라는 뜻이라 `reason`으로 둔다(맥락 지시어 '여기서·방금'이 없어도) → 워커 미발동. 새 외부 대상의 사실을 물으면 `question`. 이 ④가 빠지면 "네 의견 달라"가 `question`으로 새 리서치를 돌린다 — segment가 프레이밍을 보존해야(§3.1) classify가 이 신호를 본다.

### 3.3 `correction_node` (`nodes/correction.py`)

`utterance_types`에 `correction` 있는 세그먼트만 모아 LLM이 **슬롯 갱신 액션** 결정.
- `clear`: 슬롯 비움 + correction_log 적재.
- `replace`: `new_value`로 교체 + `source_label=USER` + log 적재.
- `ignore`: 모호하면 패스.
- 슬롯명은 10개 화이트리스트 검증. 타겟 없으면 첫 액션 슬롯을 세그먼트에 표시.
- `_CORR_SYSTEM`도 `slot_guide_text()`를 임베드 — 정정 시 슬롯 매칭이 fill·segment와 같은 정의·경계를 쓴다.

### 3.4 `extract_slot_fills_node` (`nodes/correction.py`)

dispatch 경로에서만 실행 (그래프상 dispatch 다음). 세그먼트에서 슬롯 값을 추출해 주입/확인하고, **슬롯 선택의 단일 권위**다(segment는 더 이상 힌트를 주지 않으므로 — 정의·경계만으로 정한다).
- 후보 = `claim` 라벨 가진 세그먼트. (빈 슬롯이 없어도 충돌 교체 감지를 위해 호출한다 — 예전의 "빈 슬롯 없으면 스킵"은 제거.)
- `_FILL_SYSTEM`에 `slot_guide_text()` + **`[직전 대화]`(recent_history)** + `[현재 슬롯]`·`[비어있는 슬롯]` 임베드. LLM은 fill마다 `{slot, value, kind(decision|exploration), confidence(clear|ambiguous), alt_slots[], reason, adequate}` 반환. 축: `kind`(**결정**했나)·`confidence`(**어느 슬롯**인지 명확)·`adequate`(값이 슬롯 정의의 **알맹이**를 갖췄나).
- `kind=decision` 기준: (a) 명시적 확정("X로 하자/가자/정했어") 또는 (b) 직전에 어시스턴트가 물은 슬롯 질문에 직접 답함. 단순 탐색·가설은 `exploration`. **빈 슬롯은 결정/탐색 무관하게 채우므로(가역), `kind`는 이제 주로 *이미 찬 슬롯을 덮어쓸지*(replace)를 가른다** — 확정이면 "바꿀까?" 확인, 떠보는 말이면 기존 값은 안 건드린다. (b)는 그 안전망 — `last_asked_slot`에 답한 값은 `exploration`이라도 `decision`으로 승격해, 찬 슬롯이면 교체 확인으로 간다.
- 쓰기 게이트 (**빈 슬롯 채움은 가역이라 적극적으로 채운다** — 확인은 비가역에 가까운 경우만):
  - **빈 슬롯 + 슬롯 명확 + 충분** → 즉시 주입(`source_label=USER`). **결정/탐색 무관** — 떠보는 말이어도 빈 칸이면 채운다(틀리면 "빼/바꿔"로 정정, 가역). 한 턴에 빈 슬롯이 여럿이면 **다 채운다**(턴당 캡 없음). 사용자가 자연스레 여러 사실을 한 턴에 말해도 다 박힌다.
  - **빈 슬롯 + 슬롯 애매**(`ambiguous`/`alt_slots`) → `pending_confirmations`(`confirm_kind="slot"`)에 쌓음(후보 중 빈 슬롯 없으면 스킵). 다음 턴 "어느 슬롯?" 답으로 확정 — *잘못된 칸*은 가역이 아니라 확인을 남긴다.
  - **이미 찬 슬롯 + 다른 값(결정, 정정 마커 없음)** → 덮어쓰지 않고 `pending_confirmations`(`confirm_kind="replace"`, `previous_value`=기존 값)에 쌓아 다음 턴 "바꿀까?"를 확인 — *덮어쓰기는 비가역적 손실*이라 확인 후에만. 탐색이면 기존 값은 안 건드린다. 명시적 정정("빼/말고")은 그대로 correction_node가 직접 적용(확인 불필요).
  - **공허(`adequate=false`)** → 어느 경로로도 안 채운다. "골은 결과물"처럼 슬롯 기준 미달 값이 박히는 걸 막는 충분성 게이트의 정밀 단계(coarse는 classify, §3.2). **단 조용히 버리지 않는다** — 빈 슬롯에 사용자가 직접 준 부분값이면(예: goal에 기한·실패선 없이 "10% 해소") `incomplete_fills`(매 턴 리셋)에 `{slot, partial_value}`로 남긴다. 그러면 `conversation`이 캐논 첫 빈칸 대신 **그 슬롯을 우선 되묻고**(C) **받은 부분은 인정하고 빠진 알맹이만** 콕 집어 묻는다(A, §3.7). "Goal 슬롯에 써줘"라고 칸을 지목했는데 값이 부족해 조용히 드롭되고 엉뚱한(캐논 첫 빈칸) 슬롯을 묻던 회귀를 막는다. 부재 서술 비-답("명시되지 않음" 류 `_NON_ANSWER_MARKERS`)은 진짜 부분값이 아니라 되묻기 신호에서도 거른다(헛 인정 방지).
  - `confirm_kind="commit"`은 이제 fill이 아니라 **`conversation`의 reason 제안**(§3.7)에서만 만든다 — fill의 *탐색→commit 큐(턴당 1건)·미응답 드롭* 경로는 제거했다(빈 슬롯 적극 채움으로 대체). "너무 안 채워줌"의 주범이 여러 사실을 한 턴에 말할 때 1건만 큐에 걸고 나머지를 드롭하던 이 경로였다.
- **근거 슬롯 태그(Part 1)**: 처리한 각 fill의 슬롯을 아직 태그 없는 `claim` 세그먼트에 순서대로 단다(`target_slot`) — segment가 더는 힌트를 주지 않으므로, `run_turn`의 evidence→슬롯 백필(§1)이 쓸 연결을 fill이 채운다.
- → "결정 안 한 값이 박히는"·"같은 내용이 다른 슬롯에"·"공허한 값이 박히는"·"찬 슬롯 새 값이 묵살되는" 문제를 경계 명문화 + 선택 단일화 + 결정/탐색/충분성/충돌 분리 + 확인으로 막는다.

### 3.5 `parallel_dispatch_workers_node` (`nodes/dispatch.py`)

**워커 라우트(`research/rag/logic_validator`)를 가진** 세그먼트를 보고 워커를 **2단계로 호출**.
> 라우트 유무로 판단 — `claim`·`question` 등 워커 라우트가 있는 세그먼트만 디스패치된다.
> interaction(`meta`·`recall`)·`correction`·명확화-only 세그먼트는 워커 라우트가 없어 제외(명확화-only 턴은 `_clarify_branch`가 dispatch 자체를 우회).

**2단계 디스패치** (리서치·RAG 병렬 → 논리검증 후속):
```python
# 1단계: 외부 사실 + 회사 문서 — 전 세그먼트 병렬, 먼저 끝난 워커부터 발행
research → run_research(subject) → report              # asyncio.as_completed; report를 idx로 보관
rag      → run_rag_check(subject) → (report, rag_result)
#   완료 즉시 emit(validation_report); 반환 리스트는 디스패치 순서로 재구성(결정론)
# 2단계: 논리검증 — 1단계 전체 완료 뒤(배리어), 같은 세그먼트의 RAG 산출물이 있는 타깃만
logic_validator → run_logic_validator(subject, rag_result)   # rag_result 있을 때만 호출; asyncio.as_completed
```
> **왜 2단계인가** — `logic_validator`(validator 엔진)는 RAG가 회수한 근거(highlight·raw_source)가
> claim을 논리적으로 지지하는지 판정하므로, 1단계 RAG 산출물 `RagExtractorResult`가 먼저 있어야
> 한다. RAG는 `(ValidationReport, RagExtractorResult)`를 돌려주고 dispatch가 그 원본을 2단계로
> 넘긴다(프론트엔 ValidationReport만 발행 — 단 사내 원문은 `report.citations`의 `raw_source`로 전달).
> 매트릭스상 logic_validator는 claim에서 rag와 동반하지만 **RAG 전용 판정**이다 — `evidence_mode`로
> rag를 끄거나(research 전용) RAG가 근거를 못 찾으면 `rag_result`가 없어 2단계에서 **제외**된다
> (불필요한 "근거 없음" 카드를 만들지 않는다). 외부 리서치 근거판단은 research 워커가 자체 agreement로 따로 낸다.
>
> **`evidence_mode` 토글(both/research/rag)** — 사용자가 프론트에서 고른 근거 출처 범위를 `run_turn`이
> state에 싣고, 1단계에서 `research`/`rag` 디스패치를 이걸로 거른다(`both`=둘 다, `research`=웹만,
> `rag`=사내문서만). `derive_routes`(classify)는 순수하게 두고 **디스패치 단계에서만** 거르며,
> `logic_validator`는 같은 idx의 `rag_result` 유무를 따라간다(research 전용이면 RAG가 없어 자연히 안 돈다). 최소 한쪽은 늘 켜진다.
>
> **발행은 완료순, 반환은 디스패치순** — 1단계는 `asyncio.as_completed`로 먼저 끝난 워커(웹/사내문서)의
> 결과 카드부터 발행해 사용자가 둘 다 끝나길 기다리지 않게 한다. 다만 `turn_validation_reports`·
> `turn_evidence`는 등장 순서로 되돌려 다운스트림을 결정론으로 유지한다(`session_evidence` 누적은
> `(subject,cluster)` 중복 제거라 순서 무관).
>
> **두 근거판단 분리**: RAG 근거판단과 외부 리서치 근거판단을 따로 낸다. `logic_validator`는 **사내 RAG 근거만**
> 판정하고(research를 섞지 않음), 외부 리서치의 근거판단은 research 워커가 자체 `agreement`로 1단계에서 따로
> 낸다(각각 별도 카드). validator 엔진(`run_validator`)은 `research_evidence`(Optional[str]) 인자를 그대로
> 두되 dispatch는 더 이상 채우지 않는다(항상 `None`).
>
> **역할 분담**: `rag`는 retrieval만(`agreement=unknown`), `logic_validator`는 판정만
> (verdict→agreement: supports→confirms / contradicts→contradicts / insufficient→partial /
> unrelated→unknown). 결과는 `turn_validation_reports`에 적재.
>
> **워커 구현 상태(실 구현)**: 세 워커 모두 실 파이프라인이다 — `research`(분해→검색→리포트),
> `rag`(rag_extractor: claim추출→폴더라우팅→Chroma 검색+하이라이트), `logic_validator`(validator
> 엔진 `run_validator`). 동기·블로킹 호출은 `asyncio.to_thread`로 감싼다. **mock 경로는 없다** —
> `OPENAI_API_KEY`가 없으면 실행 자체가 막힌다(§4).

### 3.6 출력 게이트 — 그래프 밖 (`POST /plan`)

계획서 생성은 채팅 그래프가 아니라 명시적 버튼(`POST /plan`)에서만 일어난다. 채팅(이 그래프)은
슬롯을 채우고 답할 뿐, 출력 의도를 판정하는 노드는 없다. 그 라우트가 출력 직전
`required_missing`으로 필수 슬롯(P·T·G)을 확인해 미달이면 거절(HTTP 400), 통과하면
`compose_markdown`으로 합성한다. 선택 슬롯이 비어 있으면(`optional_missing`) 계획서 버전에
`(조기 출력)`을 표기하고 빈칸은 `[미정]`으로 채운다. 프론트는 필수 슬롯이 다 차기 전엔
'계획서 생성' 버튼을 비활성화한다(2중 방어).
> `required_missing`/`optional_missing`은 순수 상태 술어라 `common/schema/state.py`에 둔다
> (plan 라우트가 import). conversation은 슬롯이 전부 차면 `deliver_plan`(ready)로 준비됐다고만 안내한다.

### 3.7 `conversation_node` (`agents/conversation/agent.py`)

state에서 **intent 목록을 결정론으로 뽑아**(`_build_intents`) **LLM 1회로 한 응답으로 렌더** (대화 에이전트). conversation_spec TRIGGER MATRIX 전체를 지원:
`ask_slot · confirm_slot · clarify · report_findings · answer_question · recall · explain_tool · reason_over_context · redirect · acknowledge · deliver_plan`.
> 구현 차이: conversation_spec은 `report_research`·`report_critique`를 별도 intent로 두지만, 코드는 한 주제의 research·rag·logic_validator 결과를 **`report_findings` 하나로 통합**해 넘긴다(렌더 프롬프트가 출처별로 구분). spec이 "둘은 한 턴에 묶일 수 있다(통합은 integrator 몫)"고 한 것을 그대로 반영.
- **intent 선택(결정론)**: 이번 턴 정정→`acknowledge`, `recall` 라벨 세그먼트→`recall`(대화 이력에서 답), `tool_help` 라벨 세그먼트→`explain_tool`(SLOT_SPECS·APP_OVERVIEW에서 답, in_scope 무관), `reason` 라벨 세그먼트→`reason_over_context`(누적 근거 `session_evidence`+대화이력을 재료로 실어 직접 추론·제안, in_scope 무관), `pending_confirmations`→`confirm_slot`(보류값과 후보 슬롯 제시; `confirm_kind="replace"`면 기존 값 `previous`를 실어 "X로 바꿀까?"로 물음), `in_scope=false`→`redirect`(단 tool_help·reason 세그먼트는 건너뛴다 — explain_tool·reason_over_context 우선), `turn_validation_reports`→주제별 `report_findings`(claim) 또는 `answer_question`(question), `clarify` 라우트→`clarify`. 위에서 막지 않았고 **확인 대기(`confirm_slot`)도 없으면** `ask_slot`(recall·explain_tool·reason_over_context·confirm_slot·clarify가 있으면 다음 질문 보류). **물을 슬롯은 `incomplete_fills`(이번 턴 사용자가 채우려다 알맹이 부족으로 못 들어간 빈 슬롯)가 있으면 그걸 우선, 없으면 `ALL_SLOTS` 첫 빈칸**(C). incomplete 슬롯이면 그 `partial_value`를 `ask_slot.partial`로 실어 **받은 부분은 인정하고 빠진 알맹이만** 묻게 한다(A — partial이 막연하면 LLM이 되풀이하지 않고 자연스레 묻는다). 빈칸이 하나도 없으면 `deliver_plan`(ready)로 준비됐다고 안내.
> **`reason_over_context`의 두 모드** — `reason` 발화가 ① **종합 요청**이면 가진 근거를 1~5문장으로 정리해 전하고, ② **제안·결정 요청**("네 생각엔 문제가 뭐야?")이면 **되묻지 않고** 그 슬롯에 들어갈 구체적 후보를 직접 제안하고 채택/수정을 한 문장으로 묻는다. 어느 슬롯인지는 코드가 주입하지 않고 LLM이 `subject`("…문제는…")와 `slots` 상태로 정한다(표현 결정은 LLM 몫). 예: 누적 근거에 "저가 커피 포화·수익성 악화"가 있을 때 `"네 생각엔 우리 문제가 뭐야?"` → **"모은 걸 보면 저가 커피 포화로 신규 점포 수익성이 떨어지는 게 핵심 문제 같아 — 이렇게 잡아볼까?"**. 근거가 비면 일반 추론으로 한 후보를 던지되 부족함을 밝히고 무엇을 먼저 찾을지 한 문장으로 잇는다. **회귀 배경**: 예전엔 제안 요청이 `question`으로 흘러 또 리서치를 돌리거나, 막판 `ask_slot`이 같은 질문("누가·어떤 상황에서…")을 사용자에게 되묻던 — "네 의견을 달라"는데 의견을 안 주는 회피였다. classify의 reason 확장(§3.2 ④) + segment의 프레이밍 보존(§3.1) + 이 제안 모드가 한 세트로 이 회피를 막는다.
> **제안→채택→슬롯반영 루프(②의 뒷단)** — 모드②에서 LLM은 메시지에 제안을 쓰면서 **구조화 필드 `proposal={slot, value, alternatives[]}`**도 함께 낸다(표현이 아니라 '무엇을 제안했나'라는 재료). `conversation_node`(코드)가 이를 받아 `pending_confirmations`에 등록한다 — 빈 슬롯이면 `confirm_kind="commit"`, 이미 찬 슬롯이면 `replace`(+`previous_value`), 여러 안을 제시하면 `candidate_values`에 [추천, *대안]을 싣는다(단일 안이면 비움). 등록은 reason intent가 있는 턴에서만, 같은 슬롯이 큐에 없을 때만(코드 가드). 그래야 다음 턴 `run_turn`이 `open_proposal`을 스냅샷하고(§2), `confirm_resolve`가 사용자의 수락(`accept`→추천안)·선택(`pick`+`chosen_index`→그 안)·거부·수정을 슬롯에 반영한다(§3.0) — `extract_slot_fills`의 commit/replace 확인과 **같은 메커니즘 재사용**(신규 분기 0). 이게 없으면 어시스턴트 제안이 메시지 텍스트로만 남아, 사용자가 "그걸로 하자"라고 해도 묶일 대상이 없고 그 발화가 `meta`/`clarify`로 떨어져 증발한다(원래 갭). 책임 경계: LLM은 제안 표현 + 제안 내용(`proposal`)만, 확인 큐 등록·commit/replace 판정·pick 해소는 코드.
- **렌더(LLM)**: intent 목록 JSON(+최근 대화 `recent_messages`)을 받아 한 메시지로 매끄럽게 연결(예: 결과 보고 → 다음 질문). `recent_messages`는 `recall` intent를 답할 때만 근거로 쓴다. 슬롯별 질문 톤은 `SLOT_SPECS[...]["question"]`(단일 원천)에서 가져와 `ask_slot.example`로 주입.
- **분류·판단은 안 함** — 무엇을 보고/질문할지는 state에서 파생, 대화는 표현만.

### 3.8 `response_integrator_node` (`nodes/integrator.py`)

**결정론 pass-through**. 응답을 만드는 일은 이제 `conversation_node`가 intent 목록으로 끝내므로,
통합기는 `pending_question`을 다시 만들지 않는다(대화 에이전트 결과를 덮어쓰지 않음).
세션 표시·디버깅용 `pending_clarifications`(이번 턴 `clarify` 세그먼트 목록)만 추려 기록.

---

## 4. LLM 호출 헬퍼 (`llm.py`)

모든 LLM 노드는 `call_json(system, user, schema, *, reasoning_effort=None)` 하나만 부른다. **mock/live 모드 분기는 없다** —
`OPENAI_API_KEY`가 없으면 `call_json`이 즉시 `RuntimeError`를 던지고 `api_server`도 기동 시점에
거부한다(fail-fast). 키가 있으면 항상 실 호출.
- `langchain-openai ChatOpenAI`의 **`with_structured_output(schema, method="function_calling")`** 가 스키마 변환·함수콜 강제·파싱·pydantic 검증을 한 번에 한다 — 예전의 수동 스키마 주입+`json.loads`+`model_validate`+수동 1회 재시도를 대체(LangChain doc가 권하는 구조화 출력 idiom). `method="function_calling"`은 Optional·default·중첩 필드 많은 스키마에 안전한 드롭인(strict `json_schema`는 그 제약과 충돌 위험).
- 전이 오류(429·timeout·5xx) 재시도는 그래프 노드의 `RetryPolicy`가 일원화한다(§2) — `call_json` 자체엔 재시도를 두지 않는다. (planner의 호출은 그래프 밖이라 전이 오류가 그대로 전파되지만, 예전 루프도 검증 오류만 잡았을 뿐 전이 오류는 전파했다 — 동작 동일.)
- **추론 강도**: 기본 모델 `gpt-5.4-mini`는 추론 모델이라 `reasoning_effort` 미설정이면 서버 기본(=medium) 추론으로 돌아 한 턴의 순차 호출(segment·classify·correction·conversation 등)이 수십 초로 쌓인다. `BPM_LLM_REASONING`(기본 `low`)으로 추론 강도를 낮춰 지연을 줄인다 — `call_json`이 추론 모델(`gpt-5`·`o`계열, `chat` 제외)일 때만 `ChatOpenAI`에 `reasoning_effort`로 넘기고, 비추론 모델엔 넘기지 않는다(`gpt-5` 비-chat은 langchain-openai가 `temperature`를 자동 제거). 노드별로 더 낮추고 싶으면 `call_json(..., reasoning_effort="minimal")` 인자로 덮는다.
- 모델 교체: `BPM_LLM_MODEL`(오케스트레이터, 기본 `gpt-5.4-mini`), `OPENAI_MODEL`(리서치/RAG, 기본 `gpt-5.4-mini`). 추론 강도: `BPM_LLM_REASONING`(기본 `low`).
- 오케스트레이터·리서치 진입·검색 프로바이더의 env 읽기는 `common/config.py` 명명 접근자
  (`orchestrator_model`·`require_openai_key`·`search_provider` 등)로 모은다 — 이 호출부는
  `os.environ`을 직접 읽지 않는다. (rag·validator 워커는 아직 `OPENAI_MODEL`·`OPENAI_API_KEY`를
  직접 읽어 이 표면 밖.) 두 모델 노브가 갈린 이유도 거기 적혀 있다.

---

## 5. 매 턴 처리 패턴 (메시지 종류별)

| 메시지 종류 | 워커 호출 | 슬롯 변경 | 분기 |
|---|---|---|---|
| 신규 단일 발화 | 라벨에 따라 | 잠재적 | 8유형 라벨링 → 매트릭스 |
| 신규 다중 발화 | 세그먼트별 병렬 | 잠재적 | 라우트별 분기 |
| 정정 신호 | (재검증 보류 — 아래 갭) | **필수** | correction_node 먼저 |
| 출력 요청("뽑아줘") | 없음 (생성은 버튼 `POST /plan`) | 없음 | 채팅엔 gate 없음 — `meta`로 흐름 |
| 스코프 밖 발화 | 없음 (리다이렉트) | 없음 | `in_scope=false` → routes none |
| 메타·단순응답 | 없음 | 없음 | interaction(`meta`) — "응"·"다음" 등 |
| 되묻기 | 없음 | 없음 | interaction(`recall`) — 대화 이력에서 답("아까 ~라며?") |
| 툴/슬롯 질문 | 없음 | 없음 | interaction(`tool_help`) — SLOT_SPECS·APP_OVERVIEW로 답("솔루션 슬롯이 뭐야?") |
| 추론·제안 요청 | 없음 | 없음 | interaction(`reason`) — 누적 근거·대화이력에서 직접 추론·제안("여기서 문제점 추론해봐", "네 생각엔 문제가 뭐야?") |
| 애매한 슬롯 값(어느 칸) | 라벨에 따라 | 보류→확인 후 | fill `ambiguous` → `confirm_slot`(slot) → 다음 턴 `confirm_resolve` |
| 탐색·미결정 값 | 라벨에 따라 | 보류→확인 후(미응답 드롭) | fill `kind=exploration` → `confirm_slot`(commit, "이거 X에 넣을까요?") |
| 확인 응답(넣어/빼/바꿔) | 없음 | confirm_resolve가 반영 | 순수 답이면 `confirmation_consumed` → `_post_confirm_branch`로 segment 이하 우회 |
| 공허한 슬롯 답("골=결과물") | 없음(되묻기) | 안 채움 → `incomplete_fills` | classify `clarification_needed`(coarse) / fill `adequate=false`(정밀) → 그 슬롯을 우선 되묻고 받은 부분은 인정(A+C) |
| 찬 슬롯과 충돌(타깃 교체) | 라벨에 따라 | 보류→교체 확인 후 | fill `confirm_kind="replace"` → `confirm_slot`("X로 바꿀까?") |

> **신호 키워드 정확도**("말고"·"빼자"·"뽑아줘")가 성능의 큰 부분. 첫 단계인
> 메시지 종류 판단이 어긋나면 그 턴 전체가 어긋난다.

### 기획서 대비 보강·갭

- **보강 (코드 > 기획서)**: 발화 유형 `question`·`recall`·`tool_help`·`reason` 추가·`opinion` 제거(→`claim` 흡수)·`meta`를 interaction tier로 — 총 8종(content 4 + interaction 4). `reason`은 종합·도출뿐 아니라 "어시스턴트에게 의견·제안을 청하는 발화"까지 흡수해, 워커 없이 누적 근거로 제안하게 한다. 슬롯 `advantage`(차별점) 추가(총 10개).
- **알려진 갭 (코드 < 기획서)**: 정정(correction) 시 교체된 슬롯 값의 **재검증 미동작**.
  기획서 5장은 리서치·RAG '재발동'을 요구하지만 현재는 슬롯 덮어쓰기만 함
  (`correction.py`의 TODO). 실 워커 연결 시 구현 예정.

---

## 6. 설계 원칙 (코드에 박힌 것)

1. **상시 진입점** — 조건부가 아니라 모든 메시지가 오케스트레이터를 거친다.
2. **판단/표현 분리** — 무엇을 물을지(오케) vs 어떻게 물을지(대화).
3. **분류 일원화** — 발화 유형(content/interaction 2-tier) 라벨링은 오케 단독. 논리검증·워커는 라벨링된 발화를 입력으로만 받음.
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
   ├─ confirm.py         제안 확인 해소 (accept/pick/reject/revise/replace + consumed 분기, START 직후)
   ├─ segment.py         세그멘테이션 + 맥락 복원 (슬롯 미선택 — 분할·복원만)
   ├─ classify.py        다중 라벨 + 라우팅 매트릭스 (대화 상태·현재 슬롯 인지)
   ├─ correction.py      정정 해소 + 슬롯 채움 (충분성·교체·애매 → pending 큐)
   ├─ dispatch.py        리서치·RAG 병렬 → 논리검증 2단계 호출
   └─ integrator.py      응답 통합 (결정론)

agents/conversation/agent.py   대화 에이전트 (intent 선택 + 한 응답 렌더)
agents/research/               리서치 실 파이프라인 (분해→검색→리포트)
agents/rag/                    RAG retrieval 워커 (rag_extractor + worker 어댑터)
agents/logic_validator/        논리검증 워커 (validator 엔진 호출 어댑터)
agents/validator/              validator 엔진 (run_validator: claim↔근거 판정)
common/schema/{state,labels}.py  슬롯·라벨·타입 정의
common/config.py               환경설정 단일 표면 (모델·키·검색 프로바이더 env 읽기)
```
