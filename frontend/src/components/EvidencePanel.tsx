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

// "https://en.wikipedia.org/wiki/2026_Ballon_d%27Or" -> "2026 Ballon d'Or"
function pageName(url: string, fallback: string): string {
  const m = url.match(/\/wiki\/([^?#]+)/);
  if (!m) return fallback;
  try {
    return decodeURIComponent(m[1]).replace(/_/g, " ");
  } catch {
    return m[1].replace(/_/g, " ");
  }
}

export default function EvidencePanel({ evidence, verifierResults }: Props) {
  // which verifier relied on which passage
  const citedBy = new Map<string, string[]>();
  for (const r of verifierResults) {
    for (const id of r.evidence_ids) citedBy.set(id, [...(citedBy.get(id) ?? []), VERIFIER_LABEL[r.verifier]]);
  }
  const sources = new Map<string, { name: string; by: Set<string> }>();
  for (const item of evidence) {
    const entry = sources.get(item.url) ?? { name: pageName(item.url, item.source), by: new Set<string>() };
    for (const v of citedBy.get(item.evidence_id) ?? []) entry.by.add(v);
    sources.set(item.url, entry);
  }
  const ordered = [...sources.entries()].sort((a, b) => b[1].by.size - a[1].by.size);
  const passages = [...evidence].sort((a, b) => (citedBy.get(b.evidence_id)?.length ?? 0) - (citedBy.get(a.evidence_id)?.length ?? 0));

  return (
    <div className="evidence-panel">
      <div className="sources" data-testid="sources">
        <h4>Verified from</h4>
        {ordered.length === 0 ? (
          <p className="muted">No source pages were found for this claim, so it could not be confirmed.</p>
        ) : (
          <ul>
            {ordered.map(([url, s]) => (
              <li key={url}>
                <a href={url} target="_blank" rel="noreferrer">
                  {s.name}
                </a>
                <span className="muted"> · Wikipedia</span>
                {s.by.size > 0 ? (
                  <span className="cited"> · used by {[...s.by].join(" + ")}</span>
                ) : (
                  <span className="muted"> · searched, not relied on</span>
                )}
              </li>
            ))}
          </ul>
        )}
      </div>

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
          {passages.map((item) => (
            <li key={item.evidence_id} data-testid="passage">
              <div className="passage-head">
                <span className="tag">{item.evidence_id}</span>
                <a href={item.url} target="_blank" rel="noreferrer">
                  {pageName(item.url, item.source)}
                </a>
                {citedBy.has(item.evidence_id) && <span className="cited">used by {citedBy.get(item.evidence_id)!.join(" + ")}</span>}
              </div>
              <Passage item={item} />
            </li>
          ))}
        </ol>
      )}
    </div>
  );
}
