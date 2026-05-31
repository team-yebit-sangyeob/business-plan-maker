from enum import Enum


class SourceLabel(str, Enum):
    # 출력 시 각 항목 옆에 붙는 출처 라벨 (기획서 3.3). 두 달 뒤 봐도 출처 구분.
    USER = "user"           # 대화에서 사용자가 직접 답함. 예: "타겟은 네이버로 가자"
    RESEARCH = "research"   # 웹 리서치 결과(출처 첨부). 근거성 슬롯(market)에만 허용 —
                            # P·T·G·결정 슬롯(solution/revenue/goal)은 user만(사용자 통제).
    EMPTY = "empty"         # 조기 출력 시 비어있는 슬롯 [미정] (필수 슬롯은 불가)

    @property
    def korean(self) -> str:
        return {
            SourceLabel.USER: "사용자 입력",
            SourceLabel.RESEARCH: "리서치 결과",
            SourceLabel.EMPTY: "[미정]",
        }[self]
