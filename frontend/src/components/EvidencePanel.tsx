import type { Evidence, VerificationResult } from "../types/trustlens";
import { VERDICT_LABEL, VERIFIER_LABEL, percent } from "./verdict";

interface Props {
  evidence: Evidence[];
  verifierResults: VerificationResult[];
}

function Passage({ item }: { item: Evidence }) {
  const span = item.highlight_span;
  const text = item.passage;
  if (!span || span.end <= span.start || span.end > text.length) {
    return <p className="passage">{text}</p>;
  }
  return (
    <p className="passage">
      {text.slice(0, span.start)}
      <mark title={`Closest match to the claim (score ${item.highlight_score?.toFixed(2)})`}>
        {text.slice(span.start, span.end)}
      </mark>
      {text.slice(span.end)}
    </p>
  );
}

export default function EvidencePanel({ evidence, verifierResults }: Props) {
  return (
    <div className="evidence-panel">
      <div className="verifiers">
        {verifierResults.length === 0 && <p className="muted">No verifier result is available for this claim.</p>}
        {verifierResults.map((r) => (
          <div key={r.verifier} className="verifier" data-testid={`verifier-${r.verifier}`}>
            <div className="verifier-head">
              <span className="verifier-name">{VERIFIER_LABEL[r.verifier]}</span>
              <span className="model">{r.model_version}</span>
            </div>
            <span className={`pill pill-${r.verdict}`}>{VERDICT_LABEL[r.verdict]}</span>
            <span className="muted"> confidence {percent(r.confidence)}</span>
            <p>{r.explanation}</p>
          </div>
        ))}
      </div>

      <h4>Evidence</h4>
      {evidence.length === 0 ? (
        <p className="muted">No evidence found for this claim.</p>
      ) : (
        <ol className="passages">
          {evidence.map((item) => (
            <li key={item.evidence_id} data-testid="passage">
              <div className="passage-head">
                <span className="tag">{item.evidence_id}</span>
                <a href={item.url} target="_blank" rel="noreferrer">
                  {item.source}
                </a>
              </div>
              <Passage item={item} />
            </li>
          ))}
        </ol>
      )}
    </div>
  );
}
