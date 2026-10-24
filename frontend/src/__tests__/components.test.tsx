import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import type { TrustLensClient } from "../api/client";
import ClaimCard from "../components/ClaimCard";
import CorrectionCard from "../components/CorrectionCard";
import Dashboard from "../components/Dashboard";
import EvidencePanel from "../components/EvidencePanel";
import sample from "../mocks/sampleAnalysis.json";
import type { AnalysisResult, ClaimDecision } from "../types/trustlens";

const analysis = sample as AnalysisResult;
const byVerdict = (v: string) => analysis.decisions.find((d) => d.final_verdict === v)!;
const claimOf = (d: ClaimDecision) => analysis.claims.find((c) => c.claim_id === d.claim_id)!;
const evidenceOf = (d: ClaimDecision) => analysis.evidence.filter((e) => e.claim_id === d.claim_id);

describe("ClaimCard", () => {
  it.each(["supported", "contradicted", "unverifiable"])("uses the %s colour class", (verdict) => {
    const d = byVerdict(verdict);
    render(<ClaimCard claim={claimOf(d)} decision={d} evidence={evidenceOf(d)} index={1} />);
    expect(screen.getByTestId("claim-card")).toHaveClass(`verdict-${verdict}`);
  });

  it("shows the disagreement badge only when review is needed", () => {
    const flagged = analysis.decisions.find((d) => d.needs_human_review)!;
    const agreed = analysis.decisions.find((d) => !d.needs_human_review)!;
    const { rerender } = render(<ClaimCard claim={claimOf(flagged)} decision={flagged} evidence={[]} index={1} />);
    expect(screen.getByText(/needs review/)).toBeInTheDocument();
    rerender(<ClaimCard claim={claimOf(agreed)} decision={agreed} evidence={[]} index={1} />);
    expect(screen.queryByText(/needs review/)).not.toBeInTheDocument();
  });
});

describe("EvidencePanel", () => {
  it("keeps the backend's order and marks the highlighted sentence", () => {
    const d = byVerdict("contradicted");
    const evidence = evidenceOf(d);
    render(<EvidencePanel evidence={evidence} verifierResults={d.verifier_results} />);
    const items = screen.getAllByTestId("passage");
    expect(items.map((li) => within(li).getByText(/^e_/).textContent)).toEqual(evidence.map((e) => e.evidence_id));
    const first = evidence[0];
    const span = first.highlight_span!;
    expect(items[0].querySelector("mark")?.textContent).toBe(first.passage.slice(span.start, span.end));
    expect(screen.getByTestId("verifier-gemini")).toBeInTheDocument();
    expect(screen.getByTestId("verifier-groq_qwen")).toBeInTheDocument();
  });

  it("says so when there is no evidence", () => {
    render(<EvidencePanel evidence={[]} verifierResults={[]} />);
    expect(screen.getByText("No evidence found for this claim.")).toBeInTheDocument();
  });
});

describe("CorrectionCard", () => {
  it("renders nothing for claims that are not contradicted", () => {
    const d = byVerdict("supported");
    const { container } = render(<CorrectionCard claim={claimOf(d)} decision={d} evidence={evidenceOf(d)} />);
    expect(container).toBeEmptyDOMElement();
  });

  it("shows a verified correction", () => {
    const d = byVerdict("contradicted");
    render(<CorrectionCard claim={claimOf(d)} decision={d} evidence={evidenceOf(d)} />);
    expect(screen.getByText(d.correction_text!)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Download as image" })).toBeInTheDocument();
  });

  it("hides a correction that failed the evidence check", () => {
    const d = { ...byVerdict("contradicted"), correction_verified: false };
    render(<CorrectionCard claim={claimOf(d)} decision={d} evidence={evidenceOf(d)} />);
    expect(screen.getByText("No verified correction available.")).toBeInTheDocument();
    expect(screen.queryByText(d.correction_text!)).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Download as image" })).not.toBeInTheDocument();
  });
});

describe("Dashboard", () => {
  function fakeClient(overrides: Partial<TrustLensClient> = {}): TrustLensClient {
    return {
      isMock: true,
      sampleIsLive: false,
      analyze: vi.fn().mockResolvedValue(analysis),
      getAnalysis: vi.fn(),
      listHistory: vi.fn(),
      ...overrides,
    };
  }

  it("calls analyze once with the typed question", async () => {
    const client = fakeClient();
    const onResult = vi.fn();
    render(<Dashboard client={client} result={null} onResult={onResult} />);
    await userEvent.type(screen.getByLabelText(/Ask a question/), "Tell me about Kalam");
    await userEvent.click(screen.getByRole("button", { name: "Check" }));
    expect(client.analyze).toHaveBeenCalledTimes(1);
    expect(client.analyze).toHaveBeenCalledWith("Tell me about Kalam");
    expect(onResult).toHaveBeenCalledWith(analysis);
  });

  it("shows the API error message, not a stack trace", async () => {
    const { ApiError } = await import("../api/client");
    const client = fakeClient({ analyze: vi.fn().mockRejectedValue(new ApiError("X", "The server is busy.", 503)) });
    render(<Dashboard client={client} result={null} onResult={vi.fn()} />);
    await userEvent.type(screen.getByLabelText(/Ask a question/), "q");
    await userEvent.click(screen.getByRole("button", { name: "Check" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("The server is busy.");
  });

  it("renders every claim of a result", () => {
    render(<Dashboard client={fakeClient()} result={analysis} onResult={vi.fn()} />);
    expect(screen.getAllByTestId("claim-card")).toHaveLength(analysis.claims.length);
  });
});
