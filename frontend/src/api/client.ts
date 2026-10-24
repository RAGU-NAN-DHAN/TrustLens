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

  async function request<T>(path: string, init?: RequestInit): Promise<T> {
    let res: Response;
    try {
      res = await doFetch(`${root}${path}`, init);
    } catch {
      throw new ApiError("NETWORK_ERROR", `Could not reach the TrustLens server at ${root}.`, 0);
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
        await delay(400);
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

export const client = createClient({
  baseUrl: import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000",
  useMock: (import.meta.env.VITE_USE_MOCK ?? "true") !== "false",
  sample: isLiveRun(liveSample) ? liveSample : mockSample,
});
