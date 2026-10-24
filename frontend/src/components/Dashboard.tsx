import { useState, type FormEvent } from "react";
import { ApiError, client as defaultClient, type TrustLensClient } from "../api/client";
import type { AnalysisResult } from "../types/trustlens";
import AnswerView from "./AnswerView";
import ClaimCard from "./ClaimCard";
import LensSphere from "./LensSphere";
import Overview from "./Overview";
import { percent } from "./verdict";

// grouped blue, red, white on the sphere so the slices never read as the French flag
const SPHERE_ORDER = { unverifiable: 0, contradicted: 1, supported: 2 } as const;

function sphereOrder(result: AnalysisResult | null) {
  if (!result) return [];
  return result.decisions.map((d) => d.final_verdict).sort((a, b) => SPHERE_ORDER[a] - SPHERE_ORDER[b]);
}

interface Props {
  client?: TrustLensClient;
  result: AnalysisResult | null;
  onResult: (result: AnalysisResult) => void;
  onHistory?: () => void;
  palette?: string;
}

export default function Dashboard({ client = defaultClient, result, onResult, onHistory, palette = "signal" }: Props) {
  const [question, setQuestion] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function submit(event: FormEvent) {
    event.preventDefault();
    const q = question.trim();
    if (!q || loading) return;
    setLoading(true);
    setError(null);
    try {
      onResult(await client.analyze(q));
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Something went wrong while analysing the answer.");
    } finally {
      setLoading(false);
    }
  }

  const evidenceFor = (claimId: string) => (result?.evidence ?? []).filter((e) => e.claim_id === claimId);
  const decisionFor = (claimId: string) => result?.decisions.find((d) => d.claim_id === claimId);
  const score = result?.trust_score.trust_score;
  const level = score === undefined ? "" : score >= 0.8 ? "t-supported" : score >= 0.5 ? "t-unverifiable" : "t-contradicted";

  return (
    <div className="dashboard">
      <section className="hero">
        <div className="hero-left">
          <h1 className="display headline">
            Verify any AI <span className="nowrap">answer<span className="red-dot" /></span>
          </h1>
          <p className="lede">
            Every claim is checked against evidence
            <br />
            by two independent AI verifiers.
          </p>

          <form className="ask" onSubmit={submit}>
            <label htmlFor="question" className="sr-only">
              Ask a question and TrustLens checks the AI's answer claim by claim
            </label>
            <input
              id="question"
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              placeholder="> Tell me about A. P. J. Abdul Kalam."
              autoComplete="off"
            />
            <div className="ask-actions">
              <button type="submit" className="btn-solid" disabled={loading || !question.trim()}>
                {loading ? "Checking…" : "Check"}
              </button>
              {onHistory && (
                <button type="button" className="btn-bracket" onClick={onHistory}>
                  History
                </button>
              )}
            </div>
          </form>

          {error && (
            <p className="error" role="alert">
              {error}
            </p>
          )}

          <Overview analysis={result} />
        </div>

        <div className="hero-right">
          <span className="mark tl" />
          <span className="mark tr" />
          <span className="mark bl" />
          <span className="mark br-mark" />
          <span className="mark tc" />
          <span className="mark bc" />
          <span className="red-dot rd-1" />
          <span className="red-dot rd-2" />
          <LensSphere verdicts={sphereOrder(result)} palette={palette} />
          <div className="lens-label">
            <span className="br">[</span> trust <b className={level}>{score === undefined ? "—" : percent(score)}</b>{" "}
            <span className="br">]</span>
          </div>
          <div className="lens-caption">
            {loading ? "auditing claims…" : result ? `${result.claims.length} claims · one slice each` : "idle · awaiting an answer"}
          </div>
        </div>
      </section>

      {result && (
        <section className="details">
          <div className="panel">
            <div className="panel-head">
              <span>Answer</span>
              {result.question && <span className="muted">Q: {result.question}</span>}
            </div>
            <AnswerView analysis={result} />
            <p className="muted small disclaimer">{result.trust_score.disclaimer}</p>
          </div>

          <h3 className="section-title">
            <span className="br">[</span> Claims <span className="br">]</span>
          </h3>
          {result.claims.length === 0 ? (
            <p className="muted">No factual claims were found in this answer.</p>
          ) : (
            <ol className="claims">
              {result.claims.map((claim, i) => {
                const decision = decisionFor(claim.claim_id);
                if (!decision) return null;
                return (
                  <ClaimCard
                    key={claim.claim_id}
                    claim={claim}
                    decision={decision}
                    evidence={evidenceFor(claim.claim_id)}
                    index={i + 1}
                    defaultOpen={decision.final_verdict === "contradicted"}
                  />
                );
              })}
            </ol>
          )}
        </section>
      )}
    </div>
  );
}
