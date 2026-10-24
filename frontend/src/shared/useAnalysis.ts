import { useCallback, useEffect, useState } from "react";
import { ApiError, client, type TrustLensClient } from "../api/client";
import type { AnalysisResult, HistoryItem } from "../types/trustlens";

export function useAnalysis(api: TrustLensClient = client) {
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [history, setHistory] = useState<HistoryItem[] | null>(null);

  const refreshHistory = useCallback(() => {
    api
      .listHistory()
      .then(setHistory)
      .catch(() => setHistory([]));
  }, [api]);

  useEffect(refreshHistory, [refreshHistory]);

  const analyze = useCallback(
    async (question: string) => {
      const q = question.trim();
      if (!q) return;
      setLoading(true);
      setError(null);
      try {
        setResult(await api.analyze(q));
        refreshHistory();
      } catch (err) {
        setError(err instanceof ApiError ? err.message : "Something went wrong while analysing the answer.");
      } finally {
        setLoading(false);
      }
    },
    [api, refreshHistory],
  );

  const open = useCallback(
    async (answerId: string) => {
      setError(null);
      try {
        setResult(await api.getAnalysis(answerId));
      } catch (err) {
        setError(err instanceof ApiError ? err.message : "Could not open that analysis.");
      }
    },
    [api],
  );

  return { result, loading, error, history, analyze, open, isMock: api.isMock };
}
