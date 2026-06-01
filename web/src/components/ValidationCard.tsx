import { useState } from "react";
import {
  AGREEMENT_KO,
  CLUSTER_BADGE,
  CLUSTER_LABEL,
  type ClusterName,
} from "../lib/types";

interface Props {
  cluster: ClusterName;
  subject: string;
  findings: string[];
  sources: string[];
  agreement: string;
}

export function ValidationCard({
  cluster,
  subject,
  findings,
  sources,
  agreement,
}: Props) {
  const [open, setOpen] = useState(false);
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
            <ul className="text-xs text-muted-foreground font-mono">
              {sources.map((s, i) => (
                <li key={i}>{s}</li>
              ))}
            </ul>
          </div>
        </div>
      )}
    </div>
  );
}
