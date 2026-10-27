import { useEffect, useState } from "react";
import { ApiError, client, warmUp } from "./api/client";
import Dashboard from "./components/Dashboard";
import HistoryList from "./components/HistoryList";
import type { UiName } from "./prefs";
import UiSwitch from "./shared/UiSwitch";
import type { AnalysisResult } from "./types/trustlens";

type View = "check" | "history";

interface Props {
  ui: UiName;
  onUi: (u: UiName) => void;
  theme: string;
  onTheme: (t: string) => void;
}

export default function App({ ui, onUi, theme, onTheme }: Props) {
  const [view, setView] = useState<View>("check");
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(warmUp, []);
  async function openFromHistory(answerId: string) {
    setError(null);
    try {
      setResult(await client.getAnalysis(answerId));
      setView("check");
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Could not open that analysis.");
    }
  }

  return (
    <div className="app">
      <div className="frame">
        <header className="top">
          <div className="brand display">
            TrustLens
            <span className="brand-dots" aria-hidden="true">
              <i />
              <i />
              <i />
            </span>
          </div>
          <nav>
            <UiSwitch ui={ui} onUi={onUi} theme={theme} onTheme={onTheme} className="tx-switch" />
            <button type="button" className={view === "check" ? "tab active" : "tab"} onClick={() => setView("check")}>
              Check
            </button>
            <button type="button" className={view === "history" ? "tab active" : "tab"} onClick={() => setView("history")}>
              History
            </button>
          </nav>
        </header>

        {client.isMock && (
          <p className="demo-banner">
            <span className="br">[</span> demo mode <span className="br">]</span>{" "}
            {client.sampleIsLive
              ? "real Gemini + Groq verdicts on hand-written demo claims and evidence · every question shows this run until the backend is connected"
              : "every question shows the same sample analysis · verdicts come from offline mock verifiers until the backend and API keys are connected"}
          </p>
        )}
        {error && (
          <p className="error" role="alert">
            {error}
          </p>
        )}

        <main>
          {view === "check" ? (
            <Dashboard result={result} onResult={setResult} onHistory={() => setView("history")} />
          ) : (
            <section className="history-view">
              <h2 className="display">History</h2>
              <HistoryList onOpen={openFromHistory} />
            </section>
          )}
        </main>
      </div>
    </div>
  );
}
