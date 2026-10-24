import { useEffect, useState } from "react";
import { ApiError, client as defaultClient, type TrustLensClient } from "../api/client";
import type { HistoryItem } from "../types/trustlens";
import { percent } from "./verdict";

interface Props {
  client?: TrustLensClient;
  onOpen: (answerId: string) => void;
}

export default function HistoryList({ client = defaultClient, onOpen }: Props) {
  const [items, setItems] = useState<HistoryItem[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    client
      .listHistory()
      .then((list) => !cancelled && setItems(list))
      .catch((err) => !cancelled && setError(err instanceof ApiError ? err.message : "Could not load history."));
    return () => {
      cancelled = true;
    };
  }, [client]);

  if (error) return <p className="error" role="alert">{error}</p>;
  if (!items) return <p className="muted">Loading history…</p>;
  if (items.length === 0) return <p className="muted">No analyses yet.</p>;

  return (
    <ul className="history">
      {items.map((item) => (
        <li key={item.answer_id}>
          <button type="button" onClick={() => onOpen(item.answer_id)}>
            <span className="history-q">{item.question ?? "(pasted answer)"}</span>
            <span className="muted small">
              {item.claim_count} claims · trust {percent(item.trust_score)}
              {item.created_at ? ` · ${new Date(item.created_at).toLocaleString()}` : ""}
            </span>
          </button>
        </li>
      ))}
    </ul>
  );
}
