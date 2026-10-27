// Mirrors backend/app/models/schemas.py. Change both files in the same pull request.

export type Verdict = "supported" | "contradicted" | "unverifiable";
export type VerifierName = "gemini" | "groq_qwen";

export interface Span {
  start: number;
  end: number;
}

export interface Claim {
  claim_id: string;
  answer_id: string;
  text: string;
  span: Span;
  claim_type: string;
  importance: string;
}

export interface Evidence {
  evidence_id: string;
  claim_id: string;
  source: string;
  url: string;
  passage: string;
  retrieved_at: string | null;
  keyword_score?: number | null;
  semantic_score?: number | null;
  combined_score?: number | null;
  highlight_span?: Span | null;
  highlight_score?: number | null;
}

export interface VerificationResult {
  claim_id: string;
  verifier: VerifierName;
  verdict: Verdict;
  confidence: number;
  explanation: string;
  evidence_ids: string[];
  model_version: string;
}

export interface ClaimDecision {
  claim_id: string;
  final_verdict: Verdict;
  verifier_agreement: boolean;
  verifier_results: VerificationResult[];
  self_consistency_score: number | null;
  evidence_relevance_score: number | null;
  needs_human_review: boolean;
  correction_text: string | null;
  correction_verified?: boolean | null;
  correction_support_score?: number | null;
  cache_hit: boolean;
  resolved_by_debate?: boolean;
  first_verdicts?: Record<string, Verdict> | null;
}

export interface TrustScore {
  answer_id: string;
  trust_score: number;
  method: string;
  claim_breakdown: Record<string, number>;
  disclaimer: string;
}

export interface ErrorResponse {
  error_code: string;
  message: string;
  detail: string | null;
}

export interface AnalysisResult {
  answer_id: string;
  question: string | null;
  answer_text: string;
  claims: Claim[];
  evidence: Evidence[];
  decisions: ClaimDecision[];
  trust_score: TrustScore;
  created_at: string | null;
  answer_source?: string | null;
  question_type?: string;
  notes?: string[];
}

export interface HistoryItem {
  answer_id: string;
  question: string | null;
  trust_score: number;
  claim_count: number;
  created_at: string | null;
}
