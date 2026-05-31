import { useState } from "react";

interface Props {
  subject: string;
  findings: string[];
  sources: string[];
  agreement: string;
}

export function ValidationCard({ subject, findings, sources, agreement }: Props) {
  const [open, setOpen] = useState(false);
  return (
    <div className="mt-2 border border-border rounded-md bg-muted/40">
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        className="w-full flex items-center justify-between px-3 py-2 text-left hover:bg-muted/70 transition-colors rounded-md"
      >
        <span className="text-xs font-mono uppercase tracking-wider text-muted-foreground">
          {open ? "▾" : "▸"} 검증 리포트
        </span>
        <span className="text-xs text-muted-foreground truncate ml-3">
          {subject}
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
              출처 · 일치도: {agreement}
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
