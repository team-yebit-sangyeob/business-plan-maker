import type { ChatEvent } from "./types";

/**
 * POST /chat → SSE 스트림 파싱.
 * EventSource는 GET 전용이라 fetch + ReadableStream으로 직접 파싱.
 */
export async function streamChat(
  sessionId: string,
  text: string,
  onEvent: (e: ChatEvent) => void,
  signal?: AbortSignal,
): Promise<void> {
  const res = await fetch(`/api/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId, text }),
    signal,
  });
  if (!res.ok || !res.body) {
    throw new Error(`chat stream failed: ${res.status}`);
  }
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buf = "";
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    // 일부 서버는 CRLF로 emit — LF로 정규화 후 빈 줄 기준 split
    buf += decoder.decode(value, { stream: true }).replace(/\r\n/g, "\n");
    let idx: number;
    while ((idx = buf.indexOf("\n\n")) >= 0) {
      const raw = buf.slice(0, idx);
      buf = buf.slice(idx + 2);
      const lines = raw.split("\n");
      const dataLine = lines.find((l) => l.startsWith("data:"));
      if (!dataLine) continue;
      const payload = dataLine.slice(5).trim();
      if (!payload) continue;
      // SSE event 필드까지 확인 — 백엔드의 `event: error` 프레임은 조용히 흘리지 않고
      // throw해서 호출부(catch)가 사용자에게 에러를 표시하도록 한다.
      const eventLine = lines.find((l) => l.startsWith("event:"));
      const eventName = eventLine ? eventLine.slice(6).trim() : "message";
      if (eventName === "error") {
        let detail = "응답 처리 중 오류가 발생했어요.";
        try {
          detail = JSON.parse(payload).detail ?? detail;
        } catch {
          // detail 파싱 실패 시 기본 메시지 유지
        }
        throw new Error(detail);
      }
      try {
        const evt = JSON.parse(payload) as ChatEvent;
        onEvent(evt);
      } catch {
        // 잘못된 페이로드는 무시
      }
    }
  }
}
