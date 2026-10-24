import type { AnalysisResult, Claim, ClaimDecision, Evidence, Verdict, VerifierName } from "../types/trustlens";

export const VERDICTS: Verdict[] = ["supported", "contradicted", "unverifiable"];
export const VERIFIERS: VerifierName[] = ["gemini", "groq_qwen"];

export interface ClaimRow {
  index: number;
  claim: Claim;
  decision: ClaimDecision;
  evidence: Evidence[];
  verdictOf: Partial<Record<VerifierName, Verdict>>;
  confidenceOf: Partial<Record<VerifierName, number>>;
  action: "corrected" | "review" | "no-evidence" | "no-correction" | "ok";
}

export interface Stats {
  rows: ClaimRow[];
  counts: Record<Verdict, number>;
  review: number;
  agreement: number;
  trust: number;
  evidenceCount: number;
  sourceCount: number;
  correctionsVerified: number;
  models: Partial<Record<VerifierName, string>>;
  avgConfidence: Partial<Record<VerifierName, number>>;
  verifierCounts: Record<VerifierName, Record<Verdict, number>>;
  live: boolean;
  insights: string[];
}

const emptyCounts = (): Record<Verdict, number> => ({ supported: 0, contradicted: 0, unverifiable: 0 });

function actionFor(d: ClaimDecision): ClaimRow["action"] {
  if (d.needs_human_review) return "review";
  if (d.final_verdict === "contradicted") return d.correction_text && d.correction_verified !== false ? "corrected" : "no-correction";
  if (d.final_verdict === "unverifiable") return "no-evidence";
  return "ok";
}

export const ACTION_LABEL: Record<ClaimRow["action"], string> = {
  corrected: "Corrected",
  review: "Needs review",
  "no-evidence": "No evidence",
  "no-correction": "Flagged",
  ok: "Verified",
};

export function claimNumber(i: number): string {
  return String(i).padStart(2, "0");
}

export function deriveStats(a: AnalysisResult): Stats {
  const decisions = new Map(a.decisions.map((d) => [d.claim_id, d]));
  const rows: ClaimRow[] = [];
  a.claims.forEach((claim) => {
    const decision = decisions.get(claim.claim_id);
    if (!decision) return;
    const verdictOf: ClaimRow["verdictOf"] = {};
    const confidenceOf: ClaimRow["confidenceOf"] = {};
    decision.verifier_results.forEach((r) => {
      verdictOf[r.verifier] = r.verdict;
      confidenceOf[r.verifier] = r.confidence;
    });
    rows.push({
      index: rows.length + 1,
      claim,
      decision,
      evidence: a.evidence.filter((e) => e.claim_id === claim.claim_id),
      verdictOf,
      confidenceOf,
      action: actionFor(decision),
    });
  });

  const counts = emptyCounts();
  const verifierCounts = { gemini: emptyCounts(), groq_qwen: emptyCounts() };
  const confSum: Record<VerifierName, number[]> = { gemini: [], groq_qwen: [] };
  const models: Stats["models"] = {};
  let agreed = 0;
  for (const r of rows) {
    counts[r.decision.final_verdict] += 1;
    if (r.decision.verifier_agreement) agreed += 1;
    for (const v of r.decision.verifier_results) {
      verifierCounts[v.verifier][v.verdict] += 1;
      confSum[v.verifier].push(v.confidence);
      models[v.verifier] = v.model_version;
    }
  }
  const avg = (xs: number[]) => (xs.length ? xs.reduce((s, x) => s + x, 0) / xs.length : undefined);
  const avgConfidence = { gemini: avg(confSum.gemini), groq_qwen: avg(confSum.groq_qwen) };

  const insights: string[] = [];
  for (const r of rows) {
    const n = claimNumber(r.index);
    if (r.action === "corrected") {
      insights.push(`Claim ${n} contradicts the evidence and was corrected: "${r.decision.correction_text}"`);
    } else if (r.action === "review") {
      insights.push(`The two verifiers disagree on claim ${n} ("${r.claim.text}"), so it was sent to human review.`);
    } else if (r.action === "no-evidence") {
      insights.push(`No retrieved evidence confirms or refutes claim ${n} ("${r.claim.text}").`);
    }
  }
  if (insights.length === 0 && rows.length) insights.push("Every claim in this answer is supported by the retrieved evidence.");

  return {
    rows,
    counts,
    review: rows.filter((r) => r.decision.needs_human_review).length,
    agreement: rows.length ? agreed / rows.length : 0,
    trust: a.trust_score.trust_score,
    evidenceCount: a.evidence.length,
    sourceCount: new Set(a.evidence.map((e) => e.url)).size,
    correctionsVerified: rows.filter((r) => r.action === "corrected").length,
    models,
    avgConfidence,
    verifierCounts,
    live: Object.values(models).some((m) => m && !m.startsWith("mock-")),
    insights,
  };
}
