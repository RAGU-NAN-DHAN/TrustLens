import type { Verdict, VerifierName } from "../types/trustlens";

export const VERDICT_LABEL: Record<Verdict, string> = {
  supported: "Supported",
  contradicted: "Contradicted",
  unverifiable: "Unverifiable",
};

export const VERIFIER_LABEL: Record<VerifierName, string> = {
  gemini: "Gemini",
  groq_qwen: "Groq",
};

export function percent(value: number): string {
  return `${Math.round(value * 100)}%`;
}
