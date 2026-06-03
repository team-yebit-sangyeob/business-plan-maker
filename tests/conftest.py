"""pytest 공통 설정.

agents.conversation.agent ↔ agents.orchestrator.graph 사이에 모듈 로드 순환이 있다
(agent가 call_json을, graph가 conversation_node를 모듈 최상단에서 import). conversation.agent를
먼저 import하면 부분 초기화 ImportError가 난다. orchestrator 패키지를 먼저 로드해 동작하는
순서를 박아 둔다 — 앱 진입점(api_server)도 orchestrator를 먼저 import한다.
"""
import agents.orchestrator.graph  # noqa: F401  -- 순환 import 순서 고정(먼저 로드)
