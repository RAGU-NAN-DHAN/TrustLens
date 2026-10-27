import type { AnalysisResult } from "../types/trustlens";

// The answer rebuilt from the verdicts: supported facts kept, contradicted facts replaced
// by their evidence-checked correction, unconfirmed facts kept only with a clear tag.
// Built from results the pipeline already has, so it costs no extra API calls.
export default function VerifiedAnswer({ analysis }: { analysis: AnalysisResult }) {
  const rows = analysis.claims
    .map((claim) => ({ claim, decision: analysis.decisions.find((d) => d.claim_id === claim.claim_id) }))
    .filter((r) => r.decision);
  if (rows.length === 0) return null;

  let corrected = 0;
  let removed = 0;
  let unconfirmed = 0;
  let kept = 0;
  for (const { decision } of rows) {
    if (decision!.final_verdict === "supported") kept++;
    else if (decision!.final_verdict === "contradicted" && decision!.correction_text) corrected++;
    else if (decision!.final_verdict === "contradicted") removed++;
    else unconfirmed++;
  }

  return (
    <div className="panel verified-answer" data-testid="verified-answer">
      <div className="panel-head">
        <span>Verified answer</span>
        <span className="muted">rebuilt from the evidence</span>
      </div>
      <p className="va-text">
        {rows.map(({ claim, decision }) => {
          const d = decision!;
          if (d.final_verdict === "supported") {
            return (
              <span key={claim.claim_id} className="va-ok">
                {claim.text.replace(/\.?$/, ". ")}
              </span>
            );
          }
          if (d.final_verdict === "contradicted" && d.correction_text) {
            return (
              <span key={claim.claim_id} className="va-fix" title={`Corrected. The answer said: ${claim.text}`}>
                {d.correction_text.replace(/\.?$/, ". ")}
              </span>
            );
          }
          if (d.final_verdict === "contradicted") {
            return (
              <span key={claim.claim_id} className="va-false" title="The evidence contradicts this, but no checked correction could be written">
                <s>{claim.text.replace(/\.?$/, "")}</s> <em>(false)</em>.{" "}
              </span>
            );
          }
          return (
            <span key={claim.claim_id} className="va-unsure" title="No evidence confirmed or disproved this">
              {claim.text.replace(/\.?$/, "")} <em>(unconfirmed)</em>.{" "}
            </span>
          );
        })}
      </p>
      <p className="muted small disclaimer">
        {kept} confirmed · {corrected} corrected from evidence{removed > 0 && ` · ${removed} shown false`} · {unconfirmed} unconfirmed
        {corrected > 0 && " · corrected parts are underlined; hover to see what the answer originally said"}
      </p>
    </div>
  );
}
