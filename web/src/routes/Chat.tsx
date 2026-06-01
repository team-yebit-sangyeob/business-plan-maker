import { useCallback, useEffect, useRef, useState } from "react";
import { SlotProgress } from "../components/SlotProgress";
import { MessageList } from "../components/MessageList";
import { ChatInput } from "../components/ChatInput";
import {
  createSession,
  generatePlan,
  getSession,
} from "../lib/api";
import { streamChat } from "../lib/sse";
import type {
  Message,
  PlanCard,
  SessionSnapshot,
  ChatEvent,
} from "../lib/types";
import { REQUIRED_SLOTS } from "../lib/types";

function makeId() {
  return Math.random().toString(36).slice(2, 10);
}

export default function Chat() {
  const [session, setSession] = useState<SessionSnapshot | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [streaming, setStreaming] = useState(false);
  const [building, setBuilding] = useState(false);
  const [latestPdfId, setLatestPdfId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const currentAgentId = useRef<string | null>(null);
  // 스트리밍 동안 재전송 차단(이벤트 핸들러 클로저의 stale streaming 회피용 ref).
  const streamingRef = useRef(false);

  useEffect(() => {
    createSession()
      .then(setSession)
      .catch((e) => setError(String(e)));
  }, []);

  const refreshSession = useCallback(async () => {
    if (!session) return;
    try {
      const next = await getSession(session.session_id);
      setSession(next);
    } catch (e) {
      console.warn(e);
    }
  }, [session]);

  const send = useCallback(
    async (text: string) => {
      if (!session || streamingRef.current) return;
      setError(null);

      const userMsg: Message = { id: makeId(), role: "user", text };
      const agentMsg: Message = {
        id: makeId(),
        role: "agent",
        text: "",
        activities: [],
      };
      currentAgentId.current = agentMsg.id;
      setMessages((m) => [...m, userMsg, agentMsg]);
      streamingRef.current = true;
      setStreaming(true);

      const onEvent = (e: ChatEvent) => {
        setMessages((prev) =>
          prev.map((m) => {
            if (m.id !== currentAgentId.current || m.role !== "agent") return m;
            if (e.type === "token") {
              return { ...m, text: m.text + e.text };
            }
            if (e.type === "agent_start") {
              // 새 활동 줄을 '실행 중'으로 추가
              return {
                ...m,
                activities: [
                  ...(m.activities ?? []),
                  { cluster: e.cluster, subject: e.subject, status: "running" },
                ],
              };
            }
            if (e.type === "validation_report") {
              // 같은 (cluster, subject)의 실행중 줄을 결과로 해소. 없으면 새로 추가.
              const acts = [...(m.activities ?? [])];
              const idx = acts.findIndex(
                (a) =>
                  a.status === "running" &&
                  a.cluster === e.cluster &&
                  a.subject === e.subject,
              );
              const done = {
                cluster: e.cluster,
                subject: e.subject,
                status: "done" as const,
                findings: e.findings,
                sources: e.sources,
                agreement: e.agreement,
              };
              if (idx >= 0) acts[idx] = done;
              else acts.push(done);
              return { ...m, activities: acts };
            }
            return m;
          }),
        );
        if (e.type === "slot_update" || e.type === "done") {
          // 슬롯 변경마다 가볍게 다시 조회 (서버 권위)
          refreshSession();
        }
      };

      try {
        await streamChat(session.session_id, text, onEvent);
      } catch (e) {
        setError(String(e));
      } finally {
        streamingRef.current = false;
        setStreaming(false);
        refreshSession();
      }
    },
    [session, refreshSession],
  );

  const allRequiredFilled =
    !!session &&
    REQUIRED_SLOTS.every((n) => session.slots?.[n]?.status === "filled");

  const buildPlan = useCallback(async () => {
    if (!session || !allRequiredFilled || building) return;
    setBuilding(true);
    setError(null);
    try {
      const card: PlanCard = await generatePlan(session.session_id);
      setLatestPdfId(card.plan_id);
      setMessages((m) => [
        ...m,
        {
          id: makeId(),
          role: "agent",
          text: `${card.title} 생성됨.`,
          pdf: card,
        },
      ]);
    } catch (e) {
      setError(String(e));
    } finally {
      setBuilding(false);
    }
  }, [session, allRequiredFilled, building]);

  return (
    <div className="h-full flex bg-background text-foreground">
      <SlotProgress session={session} />

      <main className="flex-1 flex flex-col">
        <header className="border-b border-border px-6 py-4 flex items-center justify-between">
          <div>
            <div className="font-mono text-[10px] uppercase tracking-wider text-muted-foreground">
              business plan agent
            </div>
            <h1 className="text-lg font-semibold mt-0.5">대화로 계획서 짜기</h1>
          </div>
          <div className="flex items-center gap-3">
            {!allRequiredFilled && (
              <span
                className="text-xs text-muted-foreground font-mono"
                title="P·T·G 필수 슬롯 통과 시 활성"
              >
                필수 슬롯 충족 시 활성
              </span>
            )}
            <button
              type="button"
              onClick={buildPlan}
              disabled={!allRequiredFilled || building}
              className="px-4 py-2 bg-primary text-primary-foreground text-sm font-medium rounded-md hover:bg-primary/90 disabled:opacity-40 transition-colors"
            >
              {building ? "생성 중…" : "계획서 생성"}
            </button>
          </div>
        </header>

        {error && (
          <div className="px-6 py-2 text-xs text-destructive font-mono border-b border-border bg-muted">
            {error}
          </div>
        )}

        <MessageList
          messages={messages}
          latestPdfId={latestPdfId}
          streaming={streaming}
        />

        <ChatInput disabled={!session || streaming} busy={streaming} onSend={send} />
      </main>
    </div>
  );
}
