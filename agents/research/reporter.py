"""리포터 — 수집 근거 + 원 claim → 하나의 검증 리포트(findings/sources/agreement).

agreement는 findings가 사용자 주장과 표면적으로 맞물리는지를 표시하는 거친 자동
플래그일 뿐 — 전제→결론 비약 같은 최종 논리 판단은 비평이 단독으로 한다.
"""
from __future__ import annotations

import json
from typing import Any

from agents.research._util import parse_json_block, traceable
from agents.research.searcher import Evidence


_VALID_AGREEMENT = ("confirms", "contradicts", "partial", "unknown")

_REPORTER_SYSTEM = """당신은 사실 검증 리포터입니다.

원래 사용자 주장(claim)과 수집된 근거(evidence)를 받아 하나의 검증 리포트를 작성하세요.

[작성 규칙]
- findings: 근거에서 확인된 핵심 사실을 한국어 bullet 2~5개로. 가능하면 구체 수치·기간·출처 맥락을 담으세요.
- sources: 근거의 출처 URL 또는 문서명을 중복 없이 나열하세요.
- agreement: 근거가 claim과 어떻게 맞물리는지 거친 플래그 하나.
    confirms = 근거가 주장을 지지
    contradicts = 근거가 주장과 반대 (예: 사용자는 "포화"라는데 데이터는 성장세)
    partial = 부분 일치하거나 근거가 혼재
    unknown = 근거 부족/무관
  (이건 표면 플래그일 뿐, 최종 논리 판단은 비평이 합니다.)
- 근거가 비어 있거나 주제와 무관하면 agreement=unknown, findings에 "외부 근거 확보 실패"를 명시하세요.

반드시 아래 JSON 형식으로만 출력하세요:
{"findings": ["..."], "sources": ["..."], "agreement": "confirms|contradicts|partial|unknown"}"""


@traceable(name="research.write_report", run_type="chain")
def write_report(
    client: Any, claim: str, evidence: list[Evidence], *, model: str
) -> dict[str, Any]:
    """근거를 종합해 {findings, sources, agreement} dict를 반환한다 (동기).

    subject·cluster는 호출자(research_main)가 결정적으로 채운다.
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
    sources = [str(s).strip() for s in (data.get("sources") or []) if str(s).strip()]
    agreement = data.get("agreement")
    if agreement not in _VALID_AGREEMENT:
        agreement = "unknown"

    return {"findings": findings, "sources": sources, "agreement": agreement}
