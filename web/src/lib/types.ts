export type SourceLabel =
  | "user"
  | "research"
  | "inference"
  | "candidate"
  | "empty";

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
  inference: "추론 도출",
  candidate: "후보 선택",
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

export type ChatEvent =
  | { type: "token"; text: string }
  | {
      type: "validation_report";
      subject: string;
      findings: string[];
      sources: string[];
      agreement: string;
    }
  | {
      type: "slot_update";
      slot: string;
      value: string | null;
      source_label: SourceLabel;
      status: Slot["status"];
    }
  | { type: "candidates"; slot: string; options: string[] }
  | { type: "done"; next_question: string; output_request: string | null };

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
      validations?: Array<{
        subject: string;
        findings: string[];
        sources: string[];
        agreement: string;
      }>;
      pdf?: PlanCard;
    };
