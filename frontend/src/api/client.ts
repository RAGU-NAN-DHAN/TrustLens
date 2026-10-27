import live from "../mocks/liveAnalysis.json";
import sample from "../mocks/sampleAnalysis.json";
import type { AnalysisResult, HistoryItem } from "../types/trustlens";

export class ApiError extends Error {
  constructor(
    public readonly error_code: string,
    message: string,
    public readonly status: number,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

export interface TrustLensClient {
  isMock: boolean;
  sampleIsLive: boolean;
  analyze(question: string): Promise<AnalysisResult>;
  getAnalysis(answerId: string): Promise<AnalysisResult>;
  listHistory(): Promise<HistoryItem[]>;
}

interface ClientOptions {
  baseUrl: string;
  useMock: boolean;
  sample?: AnalysisResult;
  fetchImpl?: typeof fetch;
}

const mockSample = sample as AnalysisResult;
// liveAnalysis.json is a real Gemini + Groq run of demo_m3.py --real on the demo claims
export const liveSample = live as AnalysisResult;
const isLiveRun = (a: AnalysisResult) => a.decisions.some((d) => d.verifier_results.some((r) => !r.model_version.startsWith("mock-")));

function copy<T>(value: T): T {
  return JSON.parse(JSON.stringify(value)) as T;
}

function delay(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function toHistoryItem(a: AnalysisResult): HistoryItem {
  return {
    answer_id: a.answer_id,
    question: a.question,
    trust_score: a.trust_score.trust_score,
    claim_count: a.claims.length,
    created_at: a.created_at,
  };
}

export function createClient({ baseUrl, useMock, fetchImpl, sample: chosen }: ClientOptions): TrustLensClient {
  const sampleAnalysis = chosen ?? mockSample;
  const doFetch = fetchImpl ?? ((input: RequestInfo | URL, init?: RequestInit) => fetch(input, init));
  const root = baseUrl.replace(/\/+$/, "");

  async function attempt(path: string, init?: RequestInit): Promise<Response> {
    // a check should never hang the page: give up after 90 s
    const ctrl = typeof AbortController !== "undefined" ? new AbortController() : null;
    const timer = ctrl ? setTimeout(() => ctrl.abort(), 90_000) : null;
    try {
      return await doFetch(`${root}${path}`, { ...init, signal: ctrl?.signal });
    } finally {
      if (timer) clearTimeout(timer);
    }
  }

  async function request<T>(path: string, init?: RequestInit): Promise<T> {
    let res: Response | null = null;
    // the free server sleeps and restarts on deploys: wait a moment and try once more before giving up
    for (let tryNo = 0; tryNo < 2 && !res; tryNo++) {
      try {
        res = await attempt(path, init);
        if (res.status === 502 || res.status === 503) {
          res = null;
          throw new Error("server restarting");
        }
      } catch {
        if (tryNo === 0) await delay(4000);
      }
    }
    if (!res) {
      throw new ApiError("NETWORK_ERROR", "The TrustLens server is waking up. Please press Check again in a few seconds.", 0);
    }
    if (!res.ok) {
      let body: { error_code?: string; message?: string } | undefined;
      try {
        body = await res.json();
      } catch {
        body = undefined;
      }
      throw new ApiError(
        body?.error_code ?? `HTTP_${res.status}`,
        body?.message ?? `The server returned status ${res.status}.`,
        res.status,
      );
    }
    return (await res.json()) as T;
  }

  if (useMock) {
    return {
      isMock: true,
      sampleIsLive: isLiveRun(sampleAnalysis),
      async analyze() {
        // long enough for the loading animation to be seen in the demo
        await delay(2600);
        return copy(sampleAnalysis);
      },
      async getAnalysis(answerId) {
        if (answerId !== sampleAnalysis.answer_id) {
          throw new ApiError("ANALYSIS_NOT_FOUND", `No analysis with id ${answerId}.`, 404);
        }
        return copy(sampleAnalysis);
      },
      async listHistory() {
        return [toHistoryItem(sampleAnalysis)];
      },
    };
  }

  return {
    isMock: false,
    sampleIsLive: false,
    analyze: (question) =>
      request<AnalysisResult>("/api/v1/analyze", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question }),
      }),
    getAnalysis: (answerId) => request<AnalysisResult>(`/api/v1/analysis/${encodeURIComponent(answerId)}`),
    listHistory: () => request<HistoryItem[]>("/api/v1/history"),
  };
}

const apiBase = (import.meta.env.VITE_API_BASE_URL ?? "").trim();
// a deployed build with no backend address falls back to demo mode instead of calling localhost
const liveWanted = (import.meta.env.VITE_USE_MOCK ?? "true") === "false";
const canGoLive = apiBase !== "" || import.meta.env.DEV;

// wake the free server as soon as the page opens, so the first check doesn't wait for a cold start
export function warmUp() {
  if (client.isMock) return;
  fetch(`${(apiBase || "http://localhost:8000").replace(/\/+$/, "")}/health`).catch(() => undefined);
}

export const client = createClient({
  baseUrl: apiBase || "http://localhost:8000",
  useMock: !(liveWanted && canGoLive),
  sample: isLiveRun(liveSample) ? liveSample : mockSample,
});
