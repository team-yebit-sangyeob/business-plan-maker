import { useEffect, useRef } from "react";
import type { Message } from "../lib/types";
import { AgentActivity, ThinkingRow } from "./AgentActivity";
import { PdfCard } from "./PdfCard";

export function MessageList({
  messages,
  latestPdfId,
  streaming,
}: {
  messages: Message[];
  latestPdfId: string | null;
  streaming: boolean;
}) {
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    ref.current?.scrollTo({ top: ref.current.scrollHeight, behavior: "smooth" });
  }, [messages]);

  return (
    <div ref={ref} className="flex-1 overflow-y-auto px-6 py-6 space-y-4">
      {messages.map((m, i) => {
        if (m.role === "user") {
          return (
            <div key={m.id} className="flex justify-end">
              <div className="max-w-[78%] bg-primary text-primary-foreground rounded-md px-3.5 py-2 text-sm whitespace-pre-wrap">
                {m.text}
              </div>
            </div>
          );
        }
        // 첫 이벤트 도착 전(아직 텍스트·활동·PDF 없음) + 마지막 메시지 + 스트리밍 중이면
        // 일반 '생각 중' 표시. agent_start/token이 오거나 스트리밍이 끝나면 자동 소멸.
        const pending =
          streaming &&
          i === messages.length - 1 &&
          !m.text &&
          !(m.activities && m.activities.length > 0) &&
          !m.pdf;
        return (
          <div key={m.id} className="flex justify-start">
            <div className="max-w-[88%] w-full">
              {pending && <ThinkingRow />}
              {m.activities && m.activities.length > 0 && (
                <AgentActivity items={m.activities} />
              )}
              {m.text && (
                <div className="mt-2 bg-muted text-foreground rounded-md px-3.5 py-2 text-sm whitespace-pre-wrap">
                  {m.text}
                </div>
              )}
              {m.pdf && (
                <PdfCard card={m.pdf} stale={latestPdfId !== m.pdf.plan_id} />
              )}
            </div>
          </div>
        );
      })}
    </div>
  );
}
