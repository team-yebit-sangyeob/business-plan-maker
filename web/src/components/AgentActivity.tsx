import { useEffect, useState } from "react";
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

/** 진행 중 경과 초 — 스피너 대신 실시간으로 올라가는 카운터(0.1초 간격 갱신).
 *  since(시작 ms)가 없으면 마운트 시점부터 잰다. */
function Elapsed({ since }: { since?: number }) {
  const [start] = useState(() => since ?? Date.now());
  const [, setTick] = useState(0);
  useEffect(() => {
    const id = setInterval(() => setTick((t) => t + 1), 100);
    return () => clearInterval(id);
  }, []);
  const secs = Math.max(0, (Date.now() - start) / 1000);
  return (
    <span
      className="shrink-0 font-mono text-[10px] tabular-nums text-muted-foreground"
      aria-label="경과 시간"
    >
      {secs.toFixed(1)}s
    </span>
  );
}

/** 진행 단계(stage) 라인 — 현재 노드 단계 라벨을 경과 초와 함께 띄운다.
 *  label이 없으면(첫 이벤트 도착 전) 기본 "생각 중…"으로 공백을 메운다. */
export function ThinkingRow({ label }: { label?: string }) {
  return (
    <div className="mt-2 flex items-center gap-2 border border-border rounded-md bg-muted/40 px-3 py-2">
      <span className="text-xs text-muted-foreground flex-1">
        {label ?? "생각 중…"}
      </span>
      <Elapsed />
    </div>
  );
}

function RunningRow({ item }: { item: Activity }) {
  return (
    <div className="flex items-center gap-2 border border-border rounded-md bg-muted/40 px-3 py-2">
      <ClusterBadge cluster={item.cluster} />
      <span className="text-xs text-muted-foreground truncate flex-1">
        {item.subject}
      </span>
      <Elapsed since={item.startedAt} />
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
            citations={a.citations ?? []}
          />
        ),
      )}
    </div>
  );
}
