import { useRef, useState } from "react";
import type { Claim, ClaimDecision, Evidence } from "../types/trustlens";

interface Props {
  claim: Claim;
  decision: ClaimDecision;
  evidence: Evidence[];
}

export default function CorrectionCard({ claim, decision, evidence }: Props) {
  const cardRef = useRef<HTMLDivElement>(null);
  const [saving, setSaving] = useState(false);

  if (decision.final_verdict !== "contradicted") return null;

  const hasCorrection = Boolean(decision.correction_text) && decision.correction_verified !== false;
  const usedIds = new Set(decision.verifier_results.flatMap((r) => r.evidence_ids));
  const sources = evidence.filter((e) => usedIds.has(e.evidence_id));
  const shown = sources.length > 0 ? sources : evidence;
  const urls = [...new Map(shown.map((e) => [e.url, e])).values()];

  async function download() {
    if (!cardRef.current) return;
    setSaving(true);
    try {
      const { default: html2canvas } = await import("html2canvas");
      const canvas = await html2canvas(cardRef.current, { backgroundColor: "#050505", scale: 2 });
      const link = document.createElement("a");
      link.download = `trustlens-correction-${claim.claim_id}.png`;
      link.href = canvas.toDataURL("image/png");
      link.click();
    } finally {
      setSaving(false);
    }
  }

  return (
    <div className="correction-wrap">
      <div className="correction-card" ref={cardRef} data-testid="correction-card">
        <div className="correction-title">TrustLens correction</div>
        <div className="correction-row">
          <span className="label label-false">Claim</span>
          <p className="false-text">{claim.text}</p>
        </div>
        <div className="correction-row">
          <span className="label label-true">Correction</span>
          {hasCorrection ? (
            <p>{decision.correction_text}</p>
          ) : (
            <p className="muted">No verified correction available.</p>
          )}
        </div>
        {decision.correction_text && decision.correction_verified === false && (
          <p className="muted small">A correction was drafted but it is not backed by the evidence, so it is not shown.</p>
        )}
        <div className="correction-row">
          <span className="label">Sources</span>
          <ul className="sources">
            {urls.map((e) => (
              <li key={e.url}>
                <a href={e.url} target="_blank" rel="noreferrer">
                  {e.url.replace(/^https?:\/\//, "")}
                </a>
              </li>
            ))}
          </ul>
        </div>
      </div>
      {hasCorrection && (
        <button type="button" className="secondary" onClick={download} disabled={saving}>
          {saving ? "Saving…" : "Download as image"}
        </button>
      )}
    </div>
  );
}
