import {
  CLUSTER_BADGE,
  CLUSTER_LABEL,
  type AgentActivity as Activity,
} from "../lib/types";
import { ValidationCard } from "./ValidationCard";

function ClusterBadge({ cluster }: { cluster: Activity["cluster"] }) {
  return (
    <span
      className={`shrink-0 px-1.5 py-0.5 rounded text-[10px] font-mono font-medium border ${CLUSTER_BADGE[cluster]}`}
    >
      {CLUSTER_LABEL[cluster]}
    </span>
  );
}

function RunningRow({ item }: { item: Activity }) {
  return (
    <div className="flex items-center gap-2 border border-border rounded-md bg-muted/40 px-3 py-2">
      <ClusterBadge cluster={item.cluster} />
      <span className="text-xs text-muted-foreground truncate flex-1">
        {item.subject}
      </span>
      <span
        className="shrink-0 w-3 h-3 rounded-full border-2 border-muted-foreground/30 border-t-muted-foreground animate-spin"
        aria-label="실행 중"
      />
    </div>
  );
}

/** 오케스트레이션이 이번 턴에 실행한 에이전트들 — 실행 중 → 결과 (클로드 도구 사용처럼). */
export function AgentActivity({ items }: { items: Activity[] }) {
  if (!items.length) return null;
  const runningCount = items.filter((a) => a.status === "running").length;
  return (
    <div className="mt-2 space-y-1.5">
      <div className="flex items-center gap-2 text-[10px] font-mono uppercase tracking-wider text-muted-foreground">
        <span>에이전트 {items.length}개</span>
        <span>·</span>
        <span>{runningCount > 0 ? `${runningCount}개 실행 중` : "완료"}</span>
      </div>
      {items.map((a, i) =>
        a.status === "running" ? (
          <RunningRow key={i} item={a} />
        ) : (
          <ValidationCard
            key={i}
            cluster={a.cluster}
            subject={a.subject}
            findings={a.findings ?? []}
            sources={a.sources ?? []}
            agreement={a.agreement ?? ""}
          />
        ),
      )}
    </div>
  );
}
