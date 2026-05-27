from enum import Enum


class SourceLabel(str, Enum):
    USER = "user"
    RESEARCH = "research"
    INFERENCE = "inference"
    CANDIDATE = "candidate"
    EMPTY = "empty"

    @property
    def korean(self) -> str:
        return {
            SourceLabel.USER: "사용자 입력",
            SourceLabel.RESEARCH: "리서치 결과",
            SourceLabel.INFERENCE: "추론 도출",
            SourceLabel.CANDIDATE: "후보 선택",
            SourceLabel.EMPTY: "[미정]",
        }[self]
