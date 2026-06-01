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
        // 응답 대기 동안 '생각 중' 표시: 마지막 에이전트 메시지에 아직 답변 텍스트·PDF가
        // 없으면 로딩을 띄운다. 워커가 '실행 중'이면 그 활동 카드(스피너)가 로딩 역할을
        // 하므로 그때만 생략 — 초기 공백과 '활동 완료~답변 시작 전' 공백을 모두 메운다.
        const acts = m.activities ?? [];
        const hasRunning = acts.some((a) => a.status === "running");
        const waitingForAnswer =
          streaming && i === messages.length - 1 && !m.text && !m.pdf;
        const showThinking = waitingForAnswer && !hasRunning;
        return (
          <div key={m.id} className="flex justify-start">
            <div className="max-w-[88%] w-full">
              {acts.length > 0 && <AgentActivity items={acts} />}
              {showThinking && <ThinkingRow />}
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
