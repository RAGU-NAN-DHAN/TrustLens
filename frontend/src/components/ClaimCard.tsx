import { useState } from "react";
import type { Claim, ClaimDecision, Evidence } from "../types/trustlens";
import CorrectionCard from "./CorrectionCard";
import EvidencePanel from "./EvidencePanel";
import { VERDICT_LABEL } from "./verdict";

interface Props {
  claim: Claim;
  decision: ClaimDecision;
  evidence: Evidence[];
  index: number;
  defaultOpen?: boolean;
}

export default function ClaimCard({ claim, decision, evidence, index, defaultOpen = false }: Props) {
  const [open, setOpen] = useState(defaultOpen);
  const verdict = decision.final_verdict;

  return (
    <li className={`claim-card verdict-${verdict}`} data-testid="claim-card" id={`claim-${claim.claim_id}`}>
      <button type="button" className="claim-head" aria-expanded={open} onClick={() => setOpen(!open)}>
        <span className="claim-index">{String(index).padStart(2, "0")}</span>
        <span className="claim-text">{claim.text}</span>
        <span className="claim-badges">
          {decision.needs_human_review && <span className="badge badge-review">Verifiers disagree · needs review</span>}
          {decision.cache_hit && <span className="badge">Cached</span>}
          {decision.resolved_by_debate && (
            <span className="badge badge-debate" title="The verifiers first disagreed, saw each other's reasoning once, and then agreed">
              Resolved by debate
            </span>
          )}
          <span className={`pill pill-${verdict}`}>{VERDICT_LABEL[verdict]}</span>
          <span className="src-toggle" title={open ? "Hide sources" : "Show where this was verified from"}>
            {open ? "−" : "+"}
          </span>
        </span>
      </button>
      {open && (
        <div className="claim-body">
          <EvidencePanel evidence={evidence} verifierResults={decision.verifier_results} />
          <CorrectionCard claim={claim} decision={decision} evidence={evidence} />
        </div>
      )}
    </li>
  );
}
