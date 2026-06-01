// 백엔드 SourceLabel(common/schema/labels.py)과 1:1 — user / research / empty 3종.
export type SourceLabel = "user" | "research" | "empty";

export interface Slot {
  value: string | null;
  source_label: SourceLabel;
  status: "empty" | "needs_clarification" | "filled";
}

// 슬롯 표시·질문 순서 = 사업계획 자연 전개 순서 (백엔드 common/schema/state.py ALL_SLOTS와 일치).
// 문제 → 고객 → 솔루션 → 시장 → 차별점 → 수익모델 → 목표 → 리소스 → 마일스톤 → 리스크.
export const ALL_SLOTS = [
  "problem",
  "target",
  "solution",
  "market",
  "advantage",
  "revenue",
  "goal",
  "resources",
  "milestones",
  "risks",
] as const;
export type SlotName = (typeof ALL_SLOTS)[number];

// 출력 게이트 필수 — 순서가 아니라 '셋 다 차야 출력' 멤버십. UI에선 굵게 표시.
export const REQUIRED_SLOTS = ["problem", "target", "goal"] as const;
const REQUIRED_SET = new Set<string>(REQUIRED_SLOTS);
export const isRequiredSlot = (name: string): boolean => REQUIRED_SET.has(name);
// 선택 = 나머지(자연 순서 유지). 카운트/문구용.
export const OPTIONAL_SLOTS = ALL_SLOTS.filter((s) => !isRequiredSlot(s));

// 질문 순서(ALL_SLOTS)대로
export const SLOT_TITLES: Record<SlotName, string> = {
  problem: "Problem",
  target: "Target",
  solution: "솔루션",
  market: "시장 근거",
  advantage: "차별점",
  revenue: "수익 모델",
  goal: "Goal",
  resources: "리소스",
  milestones: "마일스톤",
  risks: "리스크",
};

export const SOURCE_LABEL_KO: Record<SourceLabel, string> = {
  user: "사용자 입력",
  research: "리서치 결과",
  empty: "[미정]",
};

export interface SessionSnapshot {
  session_id: string;
  turn: number;
  slots: Record<string, Slot>;
  pending_question: string;
  output_request: string | null;
  correction_count?: number;
}

// 워커 클러스터 — 어느 에이전트가 냈는지 (백엔드 ValidationReport.cluster와 일치).
export type ClusterName = "research" | "rag" | "critic";

export type ChatEvent =
  | { type: "token"; text: string }
  | { type: "agent_start"; cluster: ClusterName; subject: string }
  | {
      type: "validation_report";
      cluster: ClusterName;
      subject: string;
      // 백엔드 ValidationReport는 total=False — 필드가 빠질 수 있어 옵셔널.
      findings?: string[];
      sources?: string[];
      agreement?: string;
    }
  | {
      type: "slot_update";
      slot: string;
      value: string | null;
      source_label: SourceLabel;
      status: Slot["status"];
    }
  | { type: "done"; next_question: string; output_request: string | null };

// 채팅창에 보여줄 에이전트 활동 한 줄 — 실행 중(running)으로 떴다가 결과(done)로 해소.
export interface AgentActivity {
  cluster: ClusterName;
  subject: string;
  status: "running" | "done";
  findings?: string[];
  sources?: string[];
  agreement?: string;
}

// 클러스터별 표시 라벨·뱃지 색 (채팅 활동 UI용).
export const CLUSTER_LABEL: Record<ClusterName, string> = {
  research: "웹 리서치",
  rag: "회사 문서",
  critic: "비평",
};
export const CLUSTER_BADGE: Record<ClusterName, string> = {
  research: "bg-blue-500/15 text-blue-600 border-blue-500/30",
  rag: "bg-amber-500/15 text-amber-700 border-amber-500/30",
  critic: "bg-purple-500/15 text-purple-600 border-purple-500/30",
};

// 일치도(agreement) 표시 라벨.
export const AGREEMENT_KO: Record<string, string> = {
  confirms: "근거가 뒷받침",
  contradicts: "근거와 충돌",
  partial: "부분 일치",
  unknown: "판단 보류",
};

export interface PlanCard {
  plan_id: string;
  title: string;
  pages: number;
  empty_slots: number;
  download_url: string;
  created_at: string;
}

export type Message =
  | { id: string; role: "user"; text: string }
  | {
      id: string;
      role: "agent";
      text: string;
      activities?: AgentActivity[];
      pdf?: PlanCard;
    };
