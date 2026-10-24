import type { AnalysisResult, Verdict } from "../types/trustlens";
import { VERIFIER_LABEL, percent } from "./verdict";

function Stat({ label, value, tone }: { label: string; value: string; tone?: string }) {
  return (
    <div className="stat">
      <span className="br">[</span>
      <span className="stat-label">{label}:</span>
      <span className={`stat-value ${tone ?? ""}`}>{value}</span>
      <span className="br">]</span>
    </div>
  );
}

interface Props {
  analysis: AnalysisResult | null;
  preview?: string | null;
  onPreview?: (v: Verdict) => void;
}

export default function Overview({ analysis, preview, onPreview }: Props) {
  const b = analysis?.trust_score.claim_breakdown;
  const count = (k: string) => (b ? String(b[k] ?? 0) : "—");

  return (
    <section className="overview">
      <div className="claim-strip">
        <span className="br">[</span>
        {(analysis?.decisions ?? Array.from({ length: 12 }, () => null)).map((d, i) =>
          d ? (
            <button
              key={i}
              type="button"
              className={`cell cell-${d.final_verdict} ${preview === d.final_verdict ? "cell-active" : ""}`}
              title={`Claim ${i + 1}: ${d.final_verdict} (click to preview its animation)`}
              onClick={() => onPreview?.(d.final_verdict)}
            />
          ) : (
            <span key={i} className="cell" aria-hidden="true" />
          ),
        )}
        <span className="br">]</span>
      </div>

      <div className="overview-row">
        <h2 className="display">
          Claim
          <br />
          overview
        </h2>
        <div className="stats">
          <Stat label="Claims" value={analysis ? String(analysis.claims.length) : "—"} />
          <Stat label="Supported" value={count("supported")} tone="t-supported" />
          <Stat label="Contradicted" value={count("contradicted")} tone="t-contradicted" />
          <Stat label="Unverifiable" value={count("unverifiable")} tone="t-unverifiable" />
        </div>
      </div>

      <div className="breakdown">
        <div className="breakdown-head">
          <span>Breakdown</span>
          <span className="chip">verifier confidence</span>
        </div>
        <div className="legend">
          <span>
            <span className="br">[</span> <i className="dot d-supported" /> supported <span className="br">]</span>
          </span>
          <span>
            <span className="br">[</span> <i className="dot d-contradicted" /> contradicted <span className="br">]</span>
          </span>
          <span>
            <span className="br">[</span> <i className="dot d-unverifiable" /> unverifiable <span className="br">]</span>
          </span>
        </div>
        {analysis ? (
          <>
            <div className="bars">
              {analysis.decisions.map((d) => (
                <div key={d.claim_id} className="bar-group">
                  {d.verifier_results.map((r) => (
                    <div
                      key={r.verifier}
                      className={`bar b-${r.verdict}`}
                      style={{ height: `${Math.max(4, r.confidence * 100)}%` }}
                      title={`${VERIFIER_LABEL[r.verifier]}: ${r.verdict}, ${percent(r.confidence)}`}
                    />
                  ))}
                </div>
              ))}
            </div>
            <div className="bar-axis">
              {analysis.decisions.map((d, i) => (
                <span key={d.claim_id}>{i + 1}</span>
              ))}
            </div>
            <p className="muted small">Two bars per claim: Gemini, then Groq. Height = confidence, colour = verdict.</p>
          </>
        ) : (
          <p className="muted small empty-chart">Run a check to see each claim's two verdicts here.</p>
        )}
      </div>
    </section>
  );
}
