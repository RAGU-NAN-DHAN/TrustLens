import "@fontsource/jetbrains-mono/600.css";
import { StrictMode, useEffect, useState } from "react";
import { createRoot } from "react-dom/client";
import LensScene, { type SceneMode } from "../src/components/LensScene";
import "../src/styles.css";

const MODES: [SceneMode, string][] = [
  ["idle", "Idle"],
  ["loading", "Loading (flame)"],
  ["supported", "Verified"],
  ["contradicted", "Contradicted (glitch)"],
  ["unverifiable", "Unverifiable (fog)"],
];
const THEMES: [string, string][] = [["dark", "Dark"], ["light", "Light"], ["graphite", "Graphite"]];

function Viewer() {
  const [mode, setMode] = useState<SceneMode>("idle");
  const [theme, setTheme] = useState("dark");
  const [cycle, setCycle] = useState(false);

  useEffect(() => {
    document.documentElement.dataset.palette = "heat";
    document.documentElement.dataset.theme = theme;
  }, [theme]);

  useEffect(() => {
    if (!cycle) return;
    const id = setInterval(() => {
      setMode((m) => MODES[(MODES.findIndex(([k]) => k === m) + 1) % MODES.length][0]);
    }, 4000);
    return () => clearInterval(id);
  }, [cycle]);

  return (
    <div style={{ display: "grid", gridTemplateRows: "auto 1fr", height: "100vh", background: "var(--bg)", color: "var(--text)", fontFamily: "JetBrains Mono, monospace" }}>
      <div style={{ display: "flex", flexWrap: "wrap", gap: 8, padding: 14, alignItems: "center", borderBottom: "1px solid #1e1e1e" }}>
        <b style={{ marginRight: 10 }}>TrustLens · animation test (dev only, not deployed)</b>
        {MODES.map(([k, label]) => (
          <button
            key={k}
            onClick={() => { setCycle(false); setMode(k); }}
            style={{ background: mode === k ? "#e8e8e8" : "#0c0c0c", color: mode === k ? "#000" : "#e8e8e8", border: "1px solid #2c2c2c", padding: "7px 12px", font: "inherit", cursor: "pointer" }}
          >
            {label}
          </button>
        ))}
        <label style={{ marginLeft: 12 }}>
          theme{" "}
          <select value={theme} onChange={(e) => setTheme(e.target.value)} style={{ background: "#000", color: "#e8e8e8", border: "1px solid #2c2c2c", font: "inherit" }}>
            {THEMES.map(([k, label]) => <option key={k} value={k}>{label}</option>)}
          </select>
        </label>
        <label style={{ marginLeft: 12, cursor: "pointer" }}>
          <input type="checkbox" checked={cycle} onChange={(e) => setCycle(e.target.checked)} /> auto-cycle every 4 s
        </label>
      </div>
      <div className="hero-right" style={{ position: "relative", height: "100%", maxHeight: "none", top: 0 }}>
        <LensScene mode={mode} palette="heat" />
      </div>
    </div>
  );
}

createRoot(document.getElementById("root")!).render(<StrictMode><Viewer /></StrictMode>);
