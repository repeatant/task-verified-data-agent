export type Source = { id: string; label: string; description?: string; examples: string[]; definitions?: string; schema?: Record<string, string[]>; attribution?: { title: string; url: string; license: string } };
export type Preview = { tables: Record<string, Record<string, unknown>[]> };
export type Mode = { id: string; label: string; available: boolean; reason?: string };
export type Config = { sources: Source[]; modes: Mode[]; replays: { id: string; label: string }[] };

export type Evidence = { title: string; detail: string; scope?: string; sql?: string; id?: string; rows?: unknown[]; sample?: Record<string, unknown>; versions?: Record<string, unknown>; checks?: number };
export type Revision = { target: string; before: unknown; after: unknown; reason: string; accepted?: boolean; quote?: string; evidence_ids?: string[] };
export type Result = {
  sql?: string | null;
  candidate_sql?: string | null;
  reason?: string;
  columns?: string[];
  rows?: (unknown[] | Record<string, unknown>)[];
  evidence?: Evidence[];
  revisions?: Revision[];
  finish_citations?: {signoff_ids: string[]; supporting_ids: string[]} | null;
  summary?: string | null;
  scope?: string | null;
  usage?: { requests: number; total_tokens: number; complete: boolean };
};
export type Run = {
  id: string;
  question: string;
  source_id: string;
  mode: string;
  allow_queries?: boolean;
  split_finish_evidence?: boolean;
  status: string;
  stage?: string | null;
  clarification?: string | null;
  error?: string | null;
  result?: Result | null;
  parent_id?: string | null;
  child_id?: string | null;
  definitions?: string;
};

const prefix = "/api/workbench";

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  const response = await fetch(prefix + path, { ...init, headers: { Accept: "application/json", ...init.headers } });
  if (!response.ok) {
    const body = await response.json().catch(() => null);
    const detail = body?.detail;
    const message = typeof detail === "string" ? detail : Array.isArray(detail)
      ? detail.map((entry: { msg?: string }) => entry?.msg).filter(Boolean).join("；") : null;
    throw new Error(message || `请求失败（HTTP ${response.status}）`);
  }
  return response.json() as Promise<T>;
}

function post<T>(path: string, data: unknown, key?: string) {
  return request<T>(path, {
    method: "POST",
    headers: { "Content-Type": "application/json", ...(key ? { "Idempotency-Key": key } : {}) },
    body: JSON.stringify(data),
  });
}

export type LocationRef = { kind: "run" | "replay"; id: string };
export const api = {
  preview: (id: string, signal: AbortSignal) => request<Preview>(`/sources/${encodeURIComponent(id)}/preview`, { signal }),
  config: (signal: AbortSignal) => request<Config>("/config", { signal }),
  get: (ref: LocationRef, signal: AbortSignal) => request<Run>(
    `/${ref.kind === "run" ? "runs" : "replays"}/${encodeURIComponent(ref.id)}`, { signal }),
  history: (signal: AbortSignal) => request<{ runs: Run[] }>("/runs", { signal }),
  submit: (question: string, source: string, mode: string, key: string, allowQueries?: boolean, splitEvidence?: boolean) =>
    post<Run>("/runs", { question, source_id: source, mode, allow_queries: allowQueries, split_finish_evidence: splitEvidence }, key),
  resume: (id: string) => post<Run>(`/runs/${encodeURIComponent(id)}/resume`, {}),
  clarify: (id: string, answer: string) =>
    post<Run>(`/runs/${encodeURIComponent(id)}/clarifications`, { answer }),
};

export function linkTo(ref: LocationRef) {
  return `#/${ref.kind}/${encodeURIComponent(ref.id)}`;
}

export function readLocation(hash: string): LocationRef | null {
  const match = /^#\/(run|replay)\/([^/]+)$/.exec(hash);
  if (!match) return null;
  try {
    return { kind: match[1] as LocationRef["kind"], id: decodeURIComponent(match[2]) };
  } catch {
    return null;
  }
}
