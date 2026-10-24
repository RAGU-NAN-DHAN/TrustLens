import { useMemo, useRef, useState, type FormEvent, type ReactNode } from "react";
import ClaimCard from "../components/ClaimCard";
import AnswerView from "../components/AnswerView";
import EvidencePanel from "../components/EvidencePanel";
import HistoryList from "../components/HistoryList";
import { VERDICT_LABEL, percent } from "../components/verdict";
import type { UiName } from "../prefs";
import { ACTION_LABEL, claimNumber, deriveStats, type ClaimRow, type Stats } from "../shared/derive";
import { MiniBars } from "../shared/charts";
import {
  IconArrowLeft,
  IconArrowRight,
  IconBell,
  IconChart,
  IconClock,
  IconDoc,
  IconFlow,
  IconList,
  IconLens,
  IconPlus,
  IconSearch,
  IconSettings,
  IconShield,
} from "../shared/icons";
import UiSwitch from "../shared/UiSwitch";
import { useAnalysis } from "../shared/useAnalysis";
import "./console.css";

type Tab = "overview" | "claims" | "evidence" | "history";

interface Props {
  ui: UiName;
  onUi: (u: UiName) => void;
  theme: string;
  onTheme: (t: string) => void;
}

const VAR = { supported: "var(--sup)", contradicted: "var(--con)", unverifiable: "var(--unv)" } as const;

function Kpi({ icon, title, value, unit, bars, color, foot, footValue, tone }: { icon: ReactNode; title: string; value: string; unit?: string; bars: number[]; color: string; foot: string; footValue: string; tone?: string }) {
  return (
    <div className="cx-kpi">
      <div className="cx-kpi-head">
        <span className="cx-kpi-title">
          {icon}
          {title}
        </span>
        <span className="cx-dots">···</span>
      </div>
      <div className="cx-kpi-body">
        <div className="cx-kpi-value">
          {value}
          {unit && <span>{unit}</span>}
        </div>
        <MiniBars values={bars.length ? bars : [0.2, 0.2, 0.2, 0.2, 0.2, 0.2]} color={bars.length ? color : "#3a3a3f"} />
      </div>
      <div className="cx-kpi-foot">
        <span>{foot}</span>
        <span className={tone}>{footValue}</span>
      </div>
    </div>
  );
}

function Node({ title, sub, value, badge, tone, rows }: { title: string; sub?: string; value?: string; badge: string; tone: string; rows?: [string, string][] }) {
  return (
    <div className="cx-node">
      <div className="cx-node-head">
        <span>{title}</span>
        <span className={`cx-badge ${tone}`}>
          <i />
          {badge}
        </span>
      </div>
      {sub && (
        <div className="cx-node-row">
          <span>{sub}</span>
          <b>{value}</b>
        </div>
      )}
      {rows?.map(([k, v]) => (
        <div className="cx-node-row" key={k}>
          <span>{k}</span>
          <b>{v}</b>
        </div>
      ))}
    </div>
  );
}

function Pipeline({ stats, isMock }: { stats: Stats | null; isMock: boolean }) {
  const verifierTone = stats?.live ? "ok" : "warn";
  const verifierBadge = stats ? (stats.live ? "Live" : "Mock") : "Idle";
  const gem = stats?.verifierCounts.gemini;
  const grq = stats?.verifierCounts.groq_qwen;
  const fmt = (c?: Record<string, number>) => (c ? `${c.supported}/${c.contradicted}/${c.unverifiable}` : "—");
  return (
    <div className="cx-flow">
      <div className="cx-flow-meta">{isMock ? "Demo data · hand-written claims and evidence" : "Live pipeline"}</div>
      <div className="cx-flow-grid">
        <div className="cx-row1">
          <Node title="Generate · Llama 3.3 70B" sub="Groq" value="M1" badge="Pending" tone="idle" />
        </div>
        <div className="cx-row2">
          <Node title="Split · Gemini Flash" sub="Claims" value={stats ? String(stats.rows.length) : "—"} badge={stats ? "Demo" : "Idle"} tone={stats ? "warn" : "idle"} />
        </div>
        <div className="cx-row3">
          <Node
            title="Evidence · Wikipedia + Brave"
            sub="Passages"
            value={stats ? `${stats.evidenceCount} from ${stats.sourceCount} sources` : "—"}
            badge={stats ? "Demo" : "Idle"}
            tone={stats ? "warn" : "idle"}
          />
        </div>
        <div className="cx-row4">
          <Node
            title="Verifier A · Gemini"
            badge={verifierBadge}
            tone={stats ? verifierTone : "idle"}
            rows={[
              ["Model", stats?.models.gemini ?? "gemini-2.5-pro"],
              ["S / C / U", fmt(gem)],
              ["Avg confidence", stats?.avgConfidence.gemini !== undefined ? percent(stats.avgConfidence.gemini) : "—"],
            ]}
          />
          <Node
            title="Verifier B · Groq"
            badge={verifierBadge}
            tone={stats ? verifierTone : "idle"}
            rows={[
              ["Model", stats?.models.groq_qwen ?? "non-Llama"],
              ["S / C / U", fmt(grq)],
              ["Avg confidence", stats?.avgConfidence.groq_qwen !== undefined ? percent(stats.avgConfidence.groq_qwen) : "—"],
            ]}
          />
        </div>
      </div>
    </div>
  );
}

function Setup({ stats }: { stats: Stats | null }) {
  return (
    <div className="cx-setup">
      <div className="cx-sec">1. VERIFIERS (INDEPENDENT)</div>
      <div className="cx-box">
        <div className="cx-kv">
          <span>Verifier A:</span>
          <span className="cx-tags">
            <code>{stats?.models.gemini ?? "gemini-2.5-pro"}</code>
            <b className="cx-chip">GEMINI</b>
          </span>
        </div>
        <div className="cx-kv">
          <span>Verifier B:</span>
          <span className="cx-tags">
            <code>{stats?.models.groq_qwen ?? "set GROQ_VERIFIER_MODEL"}</code>
            <b className="cx-chip">GROQ</b>
          </span>
        </div>
        <div className="cx-kv">
          <span>Grounding:</span>
          <b className="cx-chip">Evidence only (Strict)</b>
        </div>
      </div>

      <div className="cx-sec">2. CHECKS (OWN CODE)</div>
      <div className="cx-checks">
        <div className="cx-check">
          <div className="cx-check-title">
            <IconShield /> Disagreement
          </div>
          <div className="cx-kv">
            <span>Agree:</span>
            <b className="cx-chip">KEEP</b>
          </div>
          <div className="cx-kv">
            <span>Differ:</span>
            <b className="cx-chip">REVIEW</b>
          </div>
        </div>
        <div className="cx-check">
          <div className="cx-check-title">
            <IconDoc /> Correction
          </div>
          <div className="cx-kv">
            <span>Threshold:</span>
            <b className="cx-chip mono">0.50</b>
          </div>
          <div className="cx-kv">
            <span>Numbers:</span>
            <b className="cx-chip">MATCH</b>
          </div>
        </div>
        <div className="cx-check">
          <div className="cx-check-title">
            <IconSearch /> Highlight
          </div>
          <div className="cx-kv">
            <span>Min score:</span>
            <b className="cx-chip mono">0.15</b>
          </div>
          <div className="cx-kv">
            <span>Method:</span>
            <b className="cx-chip">EMBED</b>
          </div>
        </div>
      </div>

      <div className="cx-sec">3. DECISION RULE</div>
      <div className="cx-box">
        <div className="cx-rule-head">
          <span>
            Rule <b className="cx-chip mono">01</b> · <span className="cx-pill con">Disagree</span>
          </span>
        </div>
        <div className="cx-expr">
          IF [ <b>Verdict A</b> ] != [ <b>Verdict B</b> ]
        </div>
        <div className="cx-kv">
          <span>Final verdict:</span>
          <b className="cx-chip">unverifiable</b>
        </div>
        <div className="cx-kv">
          <span>Flag:</span>
          <b className="cx-chip">needs human review</b>
        </div>
      </div>
      <div className="cx-box">
        <div className="cx-rule-head">
          <span>
            Rule <b className="cx-chip mono">02</b> · <span className="cx-pill unv">Contradicted</span>
          </span>
        </div>
        <div className="cx-expr">
          IF [ <b>Final</b> ] == [ <b>contradicted</b> ]
        </div>
        <div className="cx-kv">
          <span>Then:</span>
          <b className="cx-chip">write + check correction</b>
        </div>
      </div>
    </div>
  );
}

function ExceptionTable({ rows, onPick, picked }: { rows: ClaimRow[]; onPick: (id: string) => void; picked: string | null }) {
  const exceptions = rows.filter((r) => r.action !== "ok");
  if (rows.length === 0) return <p className="cx-empty">Run a check to see flagged claims here.</p>;
  if (exceptions.length === 0) return <p className="cx-empty">No exceptions: every claim was verified.</p>;
  return (
    <table className="cx-table">
      <thead>
        <tr>
          <th>Claim</th>
          <th>Text</th>
          <th>Gemini</th>
          <th>Groq</th>
          <th className="right">Action</th>
        </tr>
      </thead>
      <tbody>
        {exceptions.map((r) => (
          <tr key={r.claim.claim_id} className={picked === r.claim.claim_id ? "picked" : ""} onClick={() => onPick(r.claim.claim_id)}>
            <td className="mono">{claimNumber(r.index)}</td>
            <td className="cx-snippet">“{r.claim.text}”</td>
            <td>{r.verdictOf.gemini ? <span className={`cx-v v-${r.verdictOf.gemini}`}>{VERDICT_LABEL[r.verdictOf.gemini]}</span> : "—"}</td>
            <td>{r.verdictOf.groq_qwen ? <span className={`cx-v v-${r.verdictOf.groq_qwen}`}>{VERDICT_LABEL[r.verdictOf.groq_qwen]}</span> : "—"}</td>
            <td className="right">
              <span className={`cx-action a-${r.action}`}>{ACTION_LABEL[r.action]}</span>
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

export default function ConsoleApp({ ui, onUi, theme, onTheme }: Props) {
  const { result, loading, error, analyze, open, isMock } = useAnalysis();
  const [question, setQuestion] = useState("");
  const [tab, setTab] = useState<Tab>("overview");
  const [picked, setPicked] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const stats = useMemo(() => (result ? deriveStats(result) : null), [result]);

  function submit(e?: FormEvent) {
    e?.preventDefault();
    if (!question.trim()) {
      inputRef.current?.focus();
      return;
    }
    setPicked(null);
    void analyze(question);
  }

  const conf = (r: ClaimRow) => ((r.confidenceOf.gemini ?? 0) + (r.confidenceOf.groq_qwen ?? 0)) / 2;
  const rows = stats?.rows ?? [];
  const pickedRow = rows.find((r) => r.claim.claim_id === picked) ?? null;

  const tabs: [Tab, string, ReactNode][] = [
    ["overview", "Overview", <IconFlow key="o" />],
    ["claims", "Claims", <IconList key="c" />],
    ["evidence", "Evidence", <IconDoc key="e" />],
    ["history", "History", <IconClock key="h" />],
  ];

  return (
    <div className="cx-root">
      <div className="cx-window">
        <aside className="cx-rail">
          <div className="cx-logo">
            <IconLens size={18} />
          </div>
          {tabs.map(([t, label, icon]) => (
            <button key={t} type="button" className={tab === t ? "active" : ""} title={label} onClick={() => setTab(t)}>
              {icon}
            </button>
          ))}
          <div className="cx-rail-sep" />
          <button type="button" title="Metrics" onClick={() => setTab("overview")}>
            <IconChart />
          </button>
          <div className="cx-rail-bottom">
            <span title="Settings">
              <IconSettings />
            </span>
          </div>
        </aside>

        <div className="cx-main">
          <header className="cx-top">
            <div className="cx-nav">
              <button type="button" aria-label="Back" onClick={() => setTab("overview")}>
                <IconArrowLeft />
              </button>
              <button type="button" aria-label="Forward" onClick={() => setTab("claims")}>
                <IconArrowRight />
              </button>
              <span className="cx-crumb">
                Audits <span>›</span> <b>{result?.question ?? "New check"}</b>
              </span>
            </div>
            <form className="cx-search" onSubmit={submit}>
              <IconSearch />
              <input
                ref={inputRef}
                id="question"
                value={question}
                onChange={(e) => setQuestion(e.target.value)}
                placeholder="Ask a question to check…"
                aria-label="Ask a question and TrustLens checks the AI's answer claim by claim"
                autoComplete="off"
              />
            </form>
            <span className="cx-icon-btn" title={isMock ? "Demo mode" : "Live"}>
              <IconBell />
              {isMock && <i />}
            </span>
            <button type="button" className="cx-primary" onClick={() => submit()} disabled={loading}>
              <IconPlus />
              {loading ? "Checking…" : "New Check"}
            </button>
          </header>

          <div className="cx-subbar">
            <div className="cx-tabs">
              {tabs.map(([t, label, icon]) => (
                <button key={t} type="button" className={tab === t ? "active" : ""} onClick={() => setTab(t)}>
                  {icon}
                  {label}
                </button>
              ))}
            </div>
            <UiSwitch ui={ui} onUi={onUi} theme={theme} onTheme={onTheme} className="cx-switch" />
          </div>

          {isMock && <p className="cx-note">Demo mode: every question loads one saved analysis (hand-written claims and evidence, real Gemini + Groq verdicts).</p>}
          {error && (
            <p className="cx-error" role="alert">
              {error}
            </p>
          )}

          {tab === "overview" && (
            <>
              <section className="cx-kpis">
                <Kpi
                  icon={<IconList />}
                  title="Claims Checked"
                  value={stats ? String(stats.rows.length) : "—"}
                  bars={rows.map(conf)}
                  color="var(--sup)"
                  foot="Verifier agreement"
                  footValue={stats ? percent(stats.agreement) : "—"}
                  tone="good"
                />
                <Kpi
                  icon={<IconShield />}
                  title="Trust Score"
                  value={stats ? String(Math.round(stats.trust * 100)) : "—"}
                  unit={stats ? "%" : undefined}
                  bars={rows.map((r) => (r.decision.final_verdict === "supported" ? 1 : r.decision.final_verdict === "unverifiable" ? 0.5 : 0.15))}
                  color="var(--unv)"
                  foot="Method"
                  footValue={result?.trust_score.method ?? "—"}
                />
                <Kpi
                  icon={<IconDoc />}
                  title="Contradicted"
                  value={stats ? String(stats.counts.contradicted) : "—"}
                  bars={rows.map((r) => (r.decision.final_verdict === "contradicted" ? 1 : 0.12))}
                  color="var(--con)"
                  foot="Corrections verified"
                  footValue={stats ? `${stats.correctionsVerified} of ${stats.counts.contradicted}` : "—"}
                  tone="good"
                />
                <Kpi
                  icon={<IconSearch />}
                  title="Needs Review"
                  value={stats ? String(stats.review) : "—"}
                  bars={rows.map((r) => (r.decision.needs_human_review ? 1 : 0.12))}
                  color="var(--con)"
                  foot="Unverifiable claims"
                  footValue={stats ? String(stats.counts.unverifiable) : "—"}
                  tone="warn"
                />
              </section>

              <section className="cx-grid">
                <div className="cx-card cx-span">
                  <div className="cx-card-head">
                    <span>
                      <IconFlow /> Pipeline Flow &amp; Health
                    </span>
                    <button type="button" className="cx-ghost" onClick={() => setTab("claims")}>
                      View Claims
                    </button>
                  </div>
                  <Pipeline stats={stats} isMock={isMock} />
                </div>

                <div className="cx-card cx-tall">
                  <div className="cx-card-head">
                    <span>
                      <IconShield /> Active Verification Setup
                    </span>
                    <span className="cx-ghost static">Contract v1</span>
                  </div>
                  <Setup stats={stats} />
                </div>

                <div className="cx-card cx-span">
                  <div className="cx-card-head">
                    <span>
                      <IconShield /> Claim Exceptions
                    </span>
                    <span className="cx-ghost static">{stats ? `${rows.filter((r) => r.action !== "ok").length} flagged` : "—"}</span>
                  </div>
                  <ExceptionTable rows={rows} picked={picked} onPick={(id) => setPicked(picked === id ? null : id)} />
                  {pickedRow && (
                    <ol className="cx-detail">
                      <ClaimCard
                        key={pickedRow.claim.claim_id}
                        claim={pickedRow.claim}
                        decision={pickedRow.decision}
                        evidence={pickedRow.evidence}
                        index={pickedRow.index}
                        defaultOpen
                      />
                    </ol>
                  )}
                </div>
              </section>

              {result && (
                <section className="cx-card cx-answer">
                  <div className="cx-card-head">
                    <span>
                      <IconDoc /> Audited Answer
                    </span>
                    <span className="cx-legend">
                      <i style={{ background: VAR.supported }} /> supported <i style={{ background: VAR.contradicted }} /> contradicted{" "}
                      <i style={{ background: VAR.unverifiable }} /> unverifiable
                    </span>
                  </div>
                  <AnswerView analysis={result} />
                  <p className="cx-fine">{result.trust_score.disclaimer}</p>
                </section>
              )}
            </>
          )}

          {tab === "claims" && (
            <section className="cx-card cx-pad">
              {rows.length === 0 ? (
                <p className="cx-empty">Run a check to see every claim.</p>
              ) : (
                <ol className="claims">
                  {rows.map((r) => (
                    <ClaimCard key={r.claim.claim_id} claim={r.claim} decision={r.decision} evidence={r.evidence} index={r.index} defaultOpen={r.decision.final_verdict === "contradicted"} />
                  ))}
                </ol>
              )}
            </section>
          )}

          {tab === "evidence" && (
            <section className="cx-card cx-pad">
              {rows.length === 0 ? (
                <p className="cx-empty">Run a check to see the evidence.</p>
              ) : (
                rows.map((r) => (
                  <div key={r.claim.claim_id} className="cx-ev">
                    <div className="cx-ev-head">
                      <span className="mono">{claimNumber(r.index)}</span> {r.claim.text}
                    </div>
                    <EvidencePanel evidence={r.evidence} verifierResults={r.decision.verifier_results} />
                  </div>
                ))
              )}
            </section>
          )}

          {tab === "history" && (
            <section className="cx-card cx-pad">
              <HistoryList
                onOpen={(id) => {
                  void open(id);
                  setTab("overview");
                }}
              />
            </section>
          )}
        </div>
      </div>
    </div>
  );
}
