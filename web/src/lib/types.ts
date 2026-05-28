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

export const REQUIRED_SLOTS = ["problem", "target", "goal"] as const;
export const OPTIONAL_SLOTS = [
  "solution",
  "advantage",
  "market",
  "revenue",
  "milestones",
  "risks",
  "resources",
] as const;
export type SlotName =
  | (typeof REQUIRED_SLOTS)[number]
  | (typeof OPTIONAL_SLOTS)[number];

export const SLOT_TITLES: Record<SlotName, string> = {
  problem: "Problem",
  target: "Target",
  goal: "Goal",
  solution: "솔루션",
  advantage: "차별점",
  market: "시장 근거",
  revenue: "수익 모델",
  milestones: "마일스톤",
  risks: "리스크",
  resources: "리소스",
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
