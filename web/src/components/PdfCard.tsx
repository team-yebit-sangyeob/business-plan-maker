import type { PlanCard } from "../lib/types";
import { planAssetUrl } from "../lib/api";

export function PdfCard({ card, stale }: { card: PlanCard; stale?: boolean }) {
  return (
    <div className="mt-3 border border-border bg-background rounded-md p-3 flex gap-3 items-center shadow-sm">
      <div className="w-12 h-16 bg-muted border border-border rounded flex items-center justify-center text-[10px] font-mono text-muted-foreground">
        PDF
      </div>
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2">
          <div className="text-sm font-semibold">{card.title}</div>
          {stale && (
            <span className="text-[10px] font-mono uppercase tracking-wider text-muted-foreground border border-border px-1.5 py-0.5 rounded">
              이전 버전
            </span>
          )}
        </div>
        <div className="text-xs text-muted-foreground mt-0.5 font-mono">
          {card.pages} 페이지
          {card.empty_slots > 0 && ` · 빈 슬롯 ${card.empty_slots}`}
          {" · "}
          {new Date(card.created_at).toLocaleTimeString()}
        </div>
      </div>
      <a
        href={planAssetUrl(card.download_url)}
        className="text-sm text-foreground underline underline-offset-4 hover:no-underline"
      >
        다운로드
      </a>
      <a
        href={planAssetUrl(card.download_url)}
        target="_blank"
        rel="noreferrer"
        className="text-sm text-muted-foreground underline underline-offset-4 hover:text-foreground"
      >
        새 탭
      </a>
    </div>
  );
}
