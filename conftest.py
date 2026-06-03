"""pytest 루트 — 저장소 루트를 sys.path에 올려 common/agents/api_server를 import 가능하게.

tests/ 아래엔 __init__.py가 없어 pytest가 테스트 파일 디렉터리만 경로에 넣는다. 이 루트 conftest
덕에 pytest가 저장소 루트도 경로에 추가한다(패키지 import 안정화).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
