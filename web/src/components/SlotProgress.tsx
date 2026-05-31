import type { SessionSnapshot, Slot } from "../lib/types";
import {
  ALL_SLOTS,
  OPTIONAL_SLOTS,
  REQUIRED_SLOTS,
  SLOT_TITLES,
  SOURCE_LABEL_KO,
  isRequiredSlot,
} from "../lib/types";

function SlotRow({
  name,
  slot,
  index,
}: {
  name: string;
  slot: Slot | undefined;
  index: number;
}) {
  const filled = slot?.status === "filled";
  const required = isRequiredSlot(name);
  return (
    <li className="flex items-center gap-2 py-1.5 text-sm">
      <span className="w-4 shrink-0 text-right font-mono text-[10px] text-muted-foreground/60">
        {index + 1}
      </span>
      <span
        className={
          filled
            ? "inline-block w-1.5 h-1.5 rounded-full bg-foreground"
            : "inline-block w-1.5 h-1.5 rounded-full border border-muted-foreground/50"
        }
      />
      <span
        className={[
          required ? "font-bold" : "font-normal",
          filled ? "text-foreground" : "text-muted-foreground",
        ].join(" ")}
      >
        {SLOT_TITLES[name as keyof typeof SLOT_TITLES] ?? name}
      </span>
      {required && (
        <span className="text-[9px] uppercase tracking-wider text-accent font-mono border border-accent/40 rounded px-1 leading-tight">
          필수
        </span>
      )}
      {filled && slot?.source_label && (
        <span className="ml-auto text-[10px] uppercase tracking-wider text-muted-foreground font-mono">
          {SOURCE_LABEL_KO[slot.source_label]}
        </span>
      )}
    </li>
  );
}

export function SlotProgress({ session }: { session: SessionSnapshot | null }) {
  const filledReq = REQUIRED_SLOTS.filter(
    (n) => session?.slots?.[n]?.status === "filled",
  ).length;
  const filledOpt = OPTIONAL_SLOTS.filter(
    (n) => session?.slots?.[n]?.status === "filled",
  ).length;

  return (
    <aside className="w-72 shrink-0 border-r border-border bg-background p-6 flex flex-col gap-6">
      <header>
        <div className="font-mono text-[10px] uppercase tracking-wider text-muted-foreground">
          slot progress
        </div>
        <h2 className="text-base font-semibold mt-1">
          질문 순서 · 필수 {REQUIRED_SLOTS.length}<span className="text-muted-foreground font-normal">(굵게)</span>
        </h2>
      </header>

      <section>
        <div className="font-mono text-[10px] uppercase tracking-wider text-muted-foreground mb-1.5">
          조사 순서대로 — 굵은 항목이 출력 필수
        </div>
        <ul>
          {ALL_SLOTS.map((n, i) => (
            <SlotRow key={n} name={n} index={i} slot={session?.slots?.[n]} />
          ))}
        </ul>
      </section>

      <footer className="mt-auto border-t border-border pt-4">
        <div className="text-xs text-muted-foreground font-mono">
          필수 {filledReq}/{REQUIRED_SLOTS.length} · 선택 {filledOpt}/{OPTIONAL_SLOTS.length}
        </div>
        <div className="mt-2 h-1 bg-muted rounded-full overflow-hidden">
          <div
            className="h-full bg-foreground rounded-full transition-all"
            style={{
              width: `${((filledReq + filledOpt) / (REQUIRED_SLOTS.length + OPTIONAL_SLOTS.length)) * 100}%`,
            }}
          />
        </div>
        {session?.correction_count ? (
          <div className="mt-3 text-[10px] uppercase tracking-wider text-muted-foreground font-mono">
            정정 {session.correction_count}회
          </div>
        ) : null}
      </footer>
    </aside>
  );
}
