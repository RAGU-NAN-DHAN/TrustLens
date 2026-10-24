import type { ReactNode } from "react";
import type { AnalysisResult } from "../types/trustlens";

interface Props {
  analysis: AnalysisResult;
}

export default function AnswerView({ analysis }: Props) {
  const verdicts = new Map(analysis.decisions.map((d) => [d.claim_id, d.final_verdict]));
  const claims = [...analysis.claims].sort((a, b) => a.span.start - b.span.start);
  const text = analysis.answer_text;

  const pieces: ReactNode[] = [];
  let cursor = 0;
  for (const claim of claims) {
    const { start, end } = claim.span;
    if (start < cursor || end > text.length || end <= start) continue;
    if (start > cursor) pieces.push(<span key={`t${cursor}`}>{text.slice(cursor, start)}</span>);
    const verdict = verdicts.get(claim.claim_id) ?? "unverifiable";
    pieces.push(
      <a key={claim.claim_id} href={`#claim-${claim.claim_id}`} className={`seg seg-${verdict}`}>
        {text.slice(start, end)}
      </a>,
    );
    cursor = end;
  }
  if (cursor < text.length) pieces.push(<span key="tail">{text.slice(cursor)}</span>);

  return <p className="answer-text">{pieces}</p>;
}
