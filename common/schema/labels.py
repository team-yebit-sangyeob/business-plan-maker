from enum import Enum


class SourceLabel(str, Enum):
    # 출력 시 각 항목 옆에 붙는 출처 라벨 (기획서 3.4). 두 달 뒤 봐도 출처 구분.
    USER = "user"           # 대화에서 사용자가 직접 답함. 예: "타겟은 네이버로 가자"
    RESEARCH = "research"   # 웹 리서치 결과(출처 첨부). 예: 시장 규모 데이터
    INFERENCE = "inference" # 다른 슬롯에서 추론 도출(검토 권장). 예: P·T·G로 만든 마일스톤
    CANDIDATE = "candidate" # 에이전트 후보 제시 → 사용자가 고른 결과. 예: 수익 모델 선택
    EMPTY = "empty"         # 조기 출력 시 비어있는 슬롯 [미정] (필수 슬롯은 불가)

    @property
    def korean(self) -> str:
        return {
            SourceLabel.USER: "사용자 입력",
            SourceLabel.RESEARCH: "리서치 결과",
            SourceLabel.INFERENCE: "추론 도출",
            SourceLabel.CANDIDATE: "후보 선택",
            SourceLabel.EMPTY: "[미정]",
        }[self]
