"""비평(Critic) — stub. 추론 점검 + 정합성 판단 (spec v0.7.5).

라벨링된 발화(`subject`) + 현재 슬롯 상태(read-only) + 1단계 산출물
(research_report·rag_context)을 입력으로 받음. dispatch가 리서치·RAG를 먼저
끝낸 뒤 그 결과를 넘긴다(critic_spec §6 "리서치·RAG 병렬 → 비평 후속").
실 로직(논리 비약·전제 충돌 탐지, severity 산정)은 추후. 분류는 일절 수행하지
않음(오케 몫).

NOTE(정합성): critic_spec은 입력 CriticInput(label·requested_modes·research_report·
rag_context …), 출력 CritiqueResult(reasoning·consistency·severity[info/warn/critical])로
정의한다. 현 stub은 시그니처(2단계 데이터 흐름)만 맞춰 두고 ValidationReport를
돌려준다 — 실 구현 시 입력/출력 스키마를 critic_spec에 맞출 것.
"""
from __future__ import annotations

from typing import Optional

from common.schema import ValidationReport
from common.schema.state import Slot


async def run_critic(
    subject: str,
    slots: dict[str, Slot],
    research_report: Optional[ValidationReport] = None,
    rag_context: Optional[ValidationReport] = None,
) -> ValidationReport:
    """추론 비약·전제 충돌 표시(막지는 않음 — 표시만). stub은 고정 응답.

    research_report·rag_context가 있으면 정합성(consistency) 비교의 근거가 되고,
    없으면 추론(reasoning) 점검만 수행한다(슬롯 상태 자체의 양립 가능성).

    실 구현 반환 예시 (가설 "일본서 통할 것" + 리서치 findings 동반):
        {
          "subject": "웹툰 IP가 일본 시장에서 통할 것이다",
          "findings": ["전제(한류·시장성장)→결론(우리 IP 통함) 사이 장르·연령 적합도 변수 누락",
                       "리서치상 한류 수용도가 30대 여성 편중 → 주장 약화"],
          "sources": [],
          "agreement": "partial",
          "cluster": "critic",
        }
    """
    received = [
        name
        for name, report in (("research", research_report), ("rag", rag_context))
        if report is not None
    ]
    mode = "정합성+추론" if received else "추론"
    basis = f" (근거 입력: {', '.join(received)})" if received else " (근거 입력 없음 — 추론만)"

    return {
        "subject": subject[:80],
        "findings": [
            f"[stub] 비평(Critic) 에이전트가 아직 연결되지 않았습니다. 모드={mode}{basis}.",
        ],
        "sources": [],
        "agreement": "unknown",
        "cluster": "critic",
    }
