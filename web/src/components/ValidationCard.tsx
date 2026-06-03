import { useState } from "react";
import {
  AGREEMENT_KO,
  CLUSTER_BADGE,
  CLUSTER_LABEL,
  type Citation,
  type ClusterName,
} from "../lib/types";

interface Props {
  cluster: ClusterName;
  subject: string;
  findings: string[];
  sources: string[];
  agreement: string;
  citations: Citation[];
}

/** score_kind별 점수 칩 라벨. none이면 빈 문자열 → 칩 생략. */
function scoreLabel(c: Citation): string {
  if (c.score_kind === "relevance")
    return `관련도 ${Math.round((c.score ?? 0) * 100)}%`;
  if (c.score_kind === "similarity_pct")
    return `유사도 ${Math.round(c.score ?? 0)}%`;
  return "";
}

/** 구조화 출처 1건 — research는 클릭 링크+관련도, rag는 파일·페이지+원문 보기 토글. */
function CitationItem({ c }: { c: Citation }) {
  const [showRaw, setShowRaw] = useState(false);
  const chip = scoreLabel(c);
  const isRag = c.cluster === "rag";
  // rag는 파일명을 제목으로, research는 제목(없으면 URL)을 제목으로.
  const title = isRag
    ? c.source_file || c.title || "(파일 미상)"
    : c.title || c.url || "(제목 없음)";
  const meta = isRag
    ? [c.page && `p.${c.page}`, c.folder].filter(Boolean).join(" · ")
    : "";
  // 원문 보기는 raw_source가 snippet보다 더 있을 때만 의미 있다.
  const hasRaw = !!(c.raw_source && c.raw_source.trim());

  return (
    <li className="border border-border rounded bg-background/60 px-2.5 py-1.5">
      <div className="flex items-start gap-2">
        {!isRag && c.url ? (
          <a
            href={c.url}
            target="_blank"
            rel="noopener noreferrer"
            className="flex-1 text-xs font-medium text-blue-600 hover:underline break-all"
          >
            {title}
          </a>
        ) : (
          <span className="flex-1 text-xs font-medium break-all">{title}</span>
        )}
        {chip && (
          <span className="shrink-0 text-[10px] font-mono px-1.5 py-0.5 rounded bg-muted text-muted-foreground">
            {chip}
          </span>
        )}
      </div>

      {isRag && meta && (
        <div className="mt-0.5 text-[10px] font-mono text-muted-foreground">
          {meta}
        </div>
      )}

      {c.snippet && (
        <p className="mt-1 text-xs text-muted-foreground leading-snug">
          {isRag && <span className="text-muted-foreground/70">하이라이트: </span>}
          “{c.snippet}”
        </p>
      )}

      {isRag && hasRaw && (
        <>
          <button
            type="button"
            onClick={() => setShowRaw((v) => !v)}
            className="mt-1 text-[10px] font-mono text-muted-foreground hover:text-foreground transition-colors"
          >
            {showRaw ? "원문 접기 ▴" : "원문 보기 ▾"}
          </button>
          {showRaw && (
            <p className="mt-1 text-xs text-muted-foreground leading-relaxed whitespace-pre-wrap border-l-2 border-border pl-2">
              {c.raw_source}
            </p>
          )}
        </>
      )}

      {!isRag && c.url && (
        <div className="mt-0.5 text-[10px] font-mono text-muted-foreground/70 break-all">
          {c.url}
        </div>
      )}
    </li>
  );
}

export function ValidationCard({
  cluster,
  subject,
  findings,
  sources,
  agreement,
  citations,
}: Props) {
  // 유효 근거(citations)가 있으면 도착 즉시 펼쳐서 상세를 보여준다(접기 가능).
  const [open, setOpen] = useState(citations.length > 0);
  return (
    <div className="border border-border rounded-md bg-muted/40">
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        className="w-full flex items-center gap-2 px-3 py-2 text-left hover:bg-muted/70 transition-colors rounded-md"
      >
        <span
          className={`shrink-0 px-1.5 py-0.5 rounded text-[10px] font-mono font-medium border ${CLUSTER_BADGE[cluster]}`}
        >
          {CLUSTER_LABEL[cluster]}
        </span>
        <span className="text-xs text-muted-foreground truncate flex-1">
          {subject}
        </span>
        <span className="shrink-0 text-xs text-muted-foreground">
          {open ? "▾" : "▸"}
        </span>
      </button>
      {open && (
        <div className="px-3 pb-3 text-sm space-y-3 border-t border-border pt-3">
          <div>
            <div className="font-mono text-[10px] uppercase tracking-wider text-muted-foreground mb-1">
              핵심 발견
            </div>
            <ul className="list-disc pl-5 space-y-0.5">
              {findings.map((f, i) => (
                <li key={i}>{f}</li>
              ))}
            </ul>
          </div>
          <div>
            <div className="font-mono text-[10px] uppercase tracking-wider text-muted-foreground mb-1">
              출처 · 일치도: {AGREEMENT_KO[agreement] ?? agreement}
            </div>
            {citations.length > 0 ? (
              <ul className="space-y-1.5">
                {citations.map((c, i) => (
                  <CitationItem key={i} c={c} />
                ))}
              </ul>
            ) : (
              <ul className="text-xs text-muted-foreground font-mono">
                {sources.map((s, i) => (
                  <li key={i}>{s}</li>
                ))}
              </ul>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
