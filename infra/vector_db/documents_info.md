# VectorDB 안내서

## 디렉토리 
```
vector_db/
├─ documents/
│  ├─ etc/
│  │  ├─ *.pdf
│  │  └─ chroma_db/
│  │
│  ├─ paper/
│  │  ├─ *.pdf
│  │  └─ chroma_db/
│  │
│  ├─ proposal/
│  │  ├─ *.pdf
│  │  └─ chroma_db/
│  │
│  └─ report/
│     ├─ *.pdf
│     └─ chroma_db/
│
├─ src/
│  └─ build_chroma_db.py
└─ documents_info.md
```

## 폴더별 설명
1. proposal : 기획서 등의 사내 문서
2. report : 사내 혹은 외부에서 발행한 보고서
3. paper : 외부에서 발표한 논문
4. etc : 이외에 스크랩된 문서 

## 문서의 신뢰도 계층
기본적으로 retrieval 시 아래 우선순위를 참고한다.

1. paper
   - 외부 논문
   - 가장 높은 신뢰도
   - 검증된 연구 결과 우선

2. report
   - 기관/기업 보고서
   - 통계 및 시장 데이터 참고

3. proposal
   - 사내 기획 문서
   - 내부 가설 및 전략 포함 가능
   - 사실 검증 필요

4. etc
   - 스크랩/참고 문서
   - 신뢰도 편차 큼
