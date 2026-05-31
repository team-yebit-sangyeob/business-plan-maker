import { useEffect, useRef } from "react";
import type { Message } from "../lib/types";
import { ValidationCard } from "./ValidationCard";
import { PdfCard } from "./PdfCard";

export function MessageList({
  messages,
  latestPdfId,
}: {
  messages: Message[];
  latestPdfId: string | null;
}) {
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    ref.current?.scrollTo({ top: ref.current.scrollHeight, behavior: "smooth" });
  }, [messages]);

  return (
    <div ref={ref} className="flex-1 overflow-y-auto px-6 py-6 space-y-4">
      {messages.map((m) => {
        if (m.role === "user") {
          return (
            <div key={m.id} className="flex justify-end">
              <div className="max-w-[78%] bg-primary text-primary-foreground rounded-md px-3.5 py-2 text-sm whitespace-pre-wrap">
                {m.text}
              </div>
            </div>
          );
        }
        return (
          <div key={m.id} className="flex justify-start">
            <div className="max-w-[88%] w-full">
              <div className="bg-muted text-foreground rounded-md px-3.5 py-2 text-sm whitespace-pre-wrap">
                {m.text}
              </div>
              {m.validations?.map((v, i) => (
                <ValidationCard key={i} {...v} />
              ))}
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
