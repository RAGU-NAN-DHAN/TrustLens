import type { AnalysisResult } from "../types/trustlens";

// Turns the backend's notes into a visible notice when an API limit was hit.
// "switched": a backup key took over, so nothing was lost.
// "degraded": a verifier or Google Search was unavailable, so some claims are marked for review.
export function limitLevel(notes: string[] | undefined): "none" | "switched" | "degraded" {
  if (!notes?.length) return "none";
  if (notes.some((n) => /unavailable/i.test(n))) return "degraded";
  if (notes.some((n) => /limit|HTTP 429|HTTP 503/i.test(n))) return "switched";
  return "none";
}

export default function LimitNotice({ analysis }: { analysis: AnalysisResult | null }) {
  const level = limitLevel(analysis?.notes);
  if (!analysis || level === "none") return null;
  const switches = analysis.notes!.filter((n) => /switched to backup key/i.test(n)).length;
  return (
    <div className={`limit-notice limit-${level}`} role="status" data-testid="limit-notice">
      <span className="battery" aria-hidden="true">
        <span className="cell" />
        <span className="cell" />
        <span className="cell" />
        <span className="cell" />
      </span>
      <div>
        <b>{level === "degraded" ? "API limit reached" : "Backup key in use"}</b>
        <p>
          {level === "degraded"
            ? "A free API limit was hit, so some checks could not run. Those claims are marked for review instead of being guessed."
            : `The main Gemini key hit its free limit; TrustLens switched to a backup key${switches > 1 ? ` (${switches} switches)` : ""} and every check still ran.`}
        </p>
      </div>
    </div>
  );
}
