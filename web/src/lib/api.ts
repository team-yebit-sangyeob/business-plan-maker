import type { PlanCard, SessionSnapshot } from "./types";

const BASE = "/api";

export async function createSession(): Promise<SessionSnapshot> {
  const res = await fetch(`${BASE}/session`, { method: "POST" });
  if (!res.ok) throw new Error("failed to create session");
  const body = await res.json();
  return body.state as SessionSnapshot;
}

export async function getSession(id: string): Promise<SessionSnapshot> {
  const res = await fetch(`${BASE}/session/${id}`);
  if (!res.ok) throw new Error("failed to fetch session");
  return (await res.json()) as SessionSnapshot;
}

export async function generatePlan(
  sessionId: string,
): Promise<PlanCard> {
  const res = await fetch(`${BASE}/plan`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId }),
  });
  if (!res.ok) {
    const detail = await res.json().catch(() => ({}));
    throw new Error(detail?.detail?.message ?? "failed to generate plan");
  }
  return (await res.json()) as PlanCard;
}

export const planAssetUrl = (path: string) => `${BASE}${path}`;
