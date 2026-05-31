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
      const dataLine = raw
        .split("\n")
        .find((l) => l.startsWith("data:"));
      if (!dataLine) continue;
      const payload = dataLine.slice(5).trim();
      if (!payload) continue;
      try {
        const evt = JSON.parse(payload) as ChatEvent;
        onEvent(evt);
      } catch {
        // 잘못된 페이로드는 무시
      }
    }
  }
}
