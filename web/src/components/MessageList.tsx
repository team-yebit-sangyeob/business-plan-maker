import { useEffect, useRef } from "react";
import ReactMarkdown from "react-markdown";
import type { Components } from "react-markdown";
import remarkGfm from "remark-gfm";
import type { Message } from "../lib/types";
import { AgentActivity, ThinkingRow } from "./AgentActivity";
import { PdfCard } from "./PdfCard";

// 봇 메시지 마크다운 렌더 — typography 플러그인 없이 요소별 Tailwind로 기존 디자인과 통일.
const MD_COMPONENTS: Components = {
  p: ({ children }) => <p className="mb-2 last:mb-0 leading-relaxed">{children}</p>,
  ul: ({ children }) => <ul className="list-disc pl-5 space-y-0.5 mb-2 last:mb-0">{children}</ul>,
  ol: ({ children }) => <ol className="list-decimal pl-5 space-y-0.5 mb-2 last:mb-0">{children}</ol>,
  li: ({ children }) => <li className="leading-relaxed">{children}</li>,
  strong: ({ children }) => <strong className="font-semibold">{children}</strong>,
  a: ({ children, href }) => (
    <a href={href} target="_blank" rel="noreferrer" className="underline underline-offset-2">
      {children}
    </a>
  ),
  code: ({ children }) => (
    <code className="px-1 py-0.5 rounded bg-background/60 font-mono text-[0.85em]">{children}</code>
  ),
};

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
              {showThinking && <ThinkingRow label={m.currentStage} />}
              {m.text && (
                <div className="mt-2 bg-muted text-foreground rounded-md px-3.5 py-2 text-sm">
                  <ReactMarkdown remarkPlugins={[remarkGfm]} components={MD_COMPONENTS}>
                    {m.text}
                  </ReactMarkdown>
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
