import VerifiedAnswer from "./VerifiedAnswer";
import LimitNotice from "./LimitNotice";
import { useState, type FormEvent } from "react";
import { ApiError, client as defaultClient, type TrustLensClient } from "../api/client";
import type { AnalysisResult } from "../types/trustlens";
import AnswerView from "./AnswerView";
import ClaimCard from "./ClaimCard";
import LensScene, { type SceneMode } from "./LensScene";
import Overview from "./Overview";
import { percent } from "./verdict";

function outcomeOf(result: AnalysisResult): SceneMode {
  const verdicts = result.decisions.map((d) => d.final_verdict);
  // no claims checked at all is not "verified"
  if (verdicts.length === 0) return "unverifiable";
  if (verdicts.includes("contradicted")) return "contradicted";
  if (result.question_type === "opinion") return "unverifiable";
  // otherwise the overall look follows the trust score: mostly confirmed reads as verified
  return result.trust_score.trust_score >= 0.75 ? "supported" : "unverifiable";
}

const SCENE_CAPTION: Record<SceneMode, string> = {
  idle: "idle · awaiting an answer",
  loading: "auditing claims…",
  supported: "every claim verified",
  contradicted: "contradiction found",
  unverifiable: "some claims cannot be verified",
};

interface Props {
  client?: TrustLensClient;
  result: AnalysisResult | null;
  onResult: (result: AnalysisResult) => void;
  onHistory?: () => void;
}

export default function Dashboard({ client = defaultClient, result, onResult, onHistory }: Props) {
  const [question, setQuestion] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [preview, setPreview] = useState<SceneMode | null>(null);

  async function submit(event: FormEvent) {
    event.preventDefault();
    const q = question.trim();
    if (!q || loading) return;
    setLoading(true);
    setError(null);
    setPreview(null);
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
  const scene: SceneMode = loading ? "loading" : !result ? "idle" : (preview ?? outcomeOf(result));
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
            <textarea
              id="question"
              rows={1}
              value={question}
              onChange={(e) => {
                setQuestion(e.target.value);
                // grow with the text so the whole question is always visible
                e.target.style.height = "auto";
                e.target.style.height = `${e.target.scrollHeight}px`;
              }}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  e.currentTarget.form?.requestSubmit();
                }
              }}
              placeholder="> Ask anything…"
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

          <LimitNotice analysis={result} />

          {error && (
            <p className="error" role="alert">
              {error}
            </p>
          )}

          <Overview analysis={result} preview={preview} onPreview={(v) => setPreview(preview === v ? null : v)} />
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
          <LensScene mode={scene} palette="heat" score={score} />
          <div className="lens-label">
            <span className="br">[</span> trust <b className={level}>{score === undefined ? "—" : percent(score)}</b>{" "}
            <span className="br">]</span>
          </div>
          <div className="lens-caption">
            {SCENE_CAPTION[scene]}
            {preview && " · preview (click the square again to reset)"}
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
            {result.question_type === "opinion" && (
              <p className="opinion-note" role="note">
                Opinion question: there is no single true answer. TrustLens checks the facts in the answer, not the opinion.
              </p>
            )}
            <AnswerView analysis={result} />
            {result.answer_source && <p className="muted small disclaimer">Answer by {result.answer_source}</p>}
            {result.notes?.map((n) => (
              <p key={n} className="muted small disclaimer">! {n}</p>
            ))}
            <p className="muted small disclaimer">{result.trust_score.disclaimer}</p>
          </div>

          <VerifiedAnswer analysis={result} />

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
