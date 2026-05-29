"""비평(Critic) — stub. 추론 점검 + 정합성 판단 (spec v0.7.5).

라벨링된 발화(`subject`)와 현재 슬롯 상태(read-only)를 입력으로 받음.
실 로직(논리 비약·전제 충돌 탐지)은 추후. 분류는 일절 수행하지 않음(오케 몫).

NOTE(정합성): critic_spec은 입력 CriticInput(label·requested_modes·research_report·
rag_context …), 출력 CritiqueResult(reasoning·consistency·severity[info/warn/critical])로
정의한다. 현 stub은 (subject, slots)만 받아 ValidationReport를 돌려준다.
특히 정합성 모드는 research_report·rag_context를 입력으로 받아야 하므로(dispatch의 NOTE
참고), 실 구현 시 dispatch를 2단계로 바꾸고 스키마를 맞출 것.
"""
from __future__ import annotations

from common.schema import ValidationReport
from common.schema.state import Slot


async def run_critic(subject: str, slots: dict[str, Slot]) -> ValidationReport:
    """추론 비약·전제 충돌 표시(막지는 않음 — 표시만). stub은 고정 응답.

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
    return {
        "subject": subject[:80],
        "findings": [
            "[stub] 비평(Critic) 에이전트가 아직 연결되지 않았습니다.",
        ],
        "sources": [],
        "agreement": "unknown",
        "cluster": "critic",
    }
