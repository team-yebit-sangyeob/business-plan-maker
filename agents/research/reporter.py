"""리포터 — 수집 근거 + 원 claim → 검증 리포트의 서술부(findings/agreement).

출처(sources/citations)는 LLM이 다시 받아쓰지 않는다 — searcher가 모은 Evidence(url/title/
snippet/score)를 research_main이 그대로 구조화 인용으로 만든다(LLM이 URL을 재타이핑하다 위조하는
경로를 끊어 출처 정확성을 보장). 그래서 리포터는 사람이 읽을 findings와 거친 일치도 플래그만 낸다.

agreement는 findings가 사용자 주장과 표면적으로 맞물리는지를 표시하는 거친 자동
플래그일 뿐 — 전제→결론 비약 같은 최종 논리 판단은 논리검증이 단독으로 한다.
"""
from __future__ import annotations

import json
from typing import Any

from agents.research._util import parse_json_block, traceable
from agents.research.searcher import Evidence


_VALID_AGREEMENT = ("confirms", "contradicts", "partial", "unknown")

_REPORTER_SYSTEM = """당신은 사실 검증 에이전트다. 스니펫 하나와 주장 하나를 받아 관계를 판정한다.

[판정 절차]
1단계: 스니펫을 읽고 핵심 내용을 파악한다. 이 단계에서 주장은 고려하지 않는다.
2단계: 주장의 핵심 내용을 파악한다.
3단계: 스니펫 내용과 주장을 비교하여 아래 기준으로 판정한다.

[판정 기준]
confirms    : 스니펫이 주장을 명확히 지지한다.
contradicts : 스니펫이 주장을 직접 반박한다.
partial     : 스니펫이 주장과 부분적으로 일치하거나 불확실하다. 지지와 반박이 혼재하거나 내용이 모호하여 판단이 어려운 경우도 partial이다.
unknown     : 스니펫의 근거가 불충분하거나 판단이 불가능하다. 스니펫이 주장과 전혀 다른 주제·범위를 다루는 경우도 unknown이다.

[판정 규칙]
주장이 맞다고 가정하지 않는다. 스니펫 내용만을 근거로 판정한다.
스니펫에 없는 내용을 추론하거나 보충하지 않는다.
스니펫이 모순되는 내용을 담고 있으면 confirms가 아닌 contradicts로 판정한다.
스니펫 내용이 모호하거나 불분명하면 unknown이 아닌 partial로 판정한다.
스니펫이 주장과 전혀 무관한 주제를 다루면 unknown으로 판정한다.

[findings 작성]
판정 근거가 된 스니펫 내 핵심 문장·수치·사실을 2~4개 추출한다.
스니펫에 없는 내용은 추가하지 않는다.

반드시 아래 JSON 형식으로만 출력한다. 다른 텍스트는 포함하지 않는다:
{"agreement": "confirms|contradicts|partial|unknown", "findings": ["근거1", "근거2"]}"""


@traceable(name="research.write_report", run_type="chain")
def write_report(
    client: Any, claim: str, evidence: list[Evidence], *, model: str
) -> dict[str, Any]:
    """근거를 종합해 {findings, agreement} dict를 반환한다 (동기).

    출처(sources/citations)는 여기서 만들지 않는다 — 호출자(research_main)가 evidence에서
    구조화 인용을 직접 만든다. subject·cluster도 호출자가 결정적으로 채운다.
    """
    user = (
        f"claim: {claim}\n"
        f"evidence:\n{json.dumps(evidence, ensure_ascii=False, indent=2)}"
    )
    resp = client.responses.create(
        model=model,
        instructions=_REPORTER_SYSTEM,
        input=[{"role": "user", "content": user}],
    )
    data = parse_json_block(resp.output_text)

    findings = [str(f).strip() for f in (data.get("findings") or []) if str(f).strip()]
    agreement = data.get("agreement")
    if agreement not in _VALID_AGREEMENT:
        agreement = "unknown"

    return {"findings": findings, "agreement": agreement}
