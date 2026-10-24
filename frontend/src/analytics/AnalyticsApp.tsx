import { useMemo, useState, type FormEvent, type ReactNode } from "react";
import AnswerView from "../components/AnswerView";
import ClaimCard from "../components/ClaimCard";
import EvidencePanel from "../components/EvidencePanel";
import { VERDICT_LABEL, percent } from "../components/verdict";
import type { UiName } from "../prefs";
import { BarChart, Donut, LineChart, Sparkline } from "../shared/charts";
import { ACTION_LABEL, claimNumber, deriveStats } from "../shared/derive";
import {
  IconArrowRight,
  IconBook,
  IconBulb,
  IconChart,
  IconCheck,
  IconClock,
  IconDoc,
  IconFlow,
  IconLayers,
  IconLens,
  IconList,
  IconPulse,
  IconSearch,
  IconSettings,
  IconShield,
} from "../shared/icons";
import UiSwitch from "../shared/UiSwitch";
import { useAnalysis } from "../shared/useAnalysis";
import "./analytics.css";

type Page = "overview" | "claims" | "evidence" | "history" | "settings";

interface Props {
  ui: UiName;
  onUi: (u: UiName) => void;
  theme: string;
  onTheme: (t: string) => void;
}

function StatCard({ icon, label, value, sub, series, tone }: { icon: ReactNode; label: string; value: string; sub: string; series: number[]; tone: "violet" | "green" | "blue" | "amber" }) {
  const color = { violet: "#8b7bff", green: "#3ddc97", blue: "#5b8cff", amber: "#ffa53d" }[tone];
  return (
    <div className={`ax-stat ax-${tone}`}>
      <div className="ax-stat-top">
        <span className="ax-stat-icon" style={{ color }}>
          {icon}
        </span>
        <div>
          <div className="ax-stat-label">{label}</div>
          <div className="ax-stat-value">{value}</div>
          <div className="ax-stat-sub">{sub}</div>
        </div>
      </div>
      <Sparkline values={series.length > 1 ? series : [0.3, 0.32, 0.31, 0.33, 0.32]} color={color} height={56} />
    </div>
  );
}

function verdictChip(v?: string) {
  if (!v) return <span className="ax-chip">—</span>;
  return <span className={`ax-chip c-${v}`}>{VERDICT_LABEL[v as keyof typeof VERDICT_LABEL]}</span>;
}

export default function AnalyticsApp({ ui, onUi, theme, onTheme }: Props) {
  const { result, loading, error, history, analyze, open, isMock } = useAnalysis();
  const [question, setQuestion] = useState("");
  const [page, setPage] = useState<Page>("overview");
  const [picked, setPicked] = useState<string | null>(null);
  const [collapsed, setCollapsed] = useState(false);
  const stats = useMemo(() => (result ? deriveStats(result) : null), [result]);
  const rows = stats?.rows ?? [];

  function submit(e: FormEvent) {
    e.preventDefault();
    if (!question.trim()) return;
    setPicked(null);
    void analyze(question);
    setPage("overview");
  }

  const nav: [Page, string, ReactNode][] = [
    ["overview", "Overview", <IconFlow key="o" />],
    ["claims", "Claims", <IconList key="c" />],
    ["evidence", "Evidence", <IconDoc key="e" />],
    ["history", "History", <IconClock key="h" />],
    ["settings", "Settings", <IconSettings key="s" />],
  ];

  const labels = rows.map((r) => claimNumber(r.index));
  const gemini = rows.map((r) => r.confidenceOf.gemini ?? 0);
  const groq = rows.map((r) => r.confidenceOf.groq_qwen ?? 0);
  const running = rows.map((_, i) => {
    const seen = rows.slice(0, i + 1);
    return seen.reduce((s, r) => s + (r.decision.final_verdict === "supported" ? 1 : r.decision.final_verdict === "unverifiable" ? 0.5 : 0), 0) / seen.length;
  });
  const evidenceScores = result?.evidence.map((e) => e.highlight_score ?? 0) ?? [];
  const evidenceLabels = result?.evidence.map((e) => e.evidence_id.replace(/^e_0*/, "").replace("_", ".")) ?? [];
  const pickedRow = rows.find((r) => r.claim.claim_id === picked) ?? null;

  const donut = stats
    ? [
        { label: "Supported", value: stats.counts.supported, color: "var(--sup)" },
        { label: "Contradicted", value: stats.counts.contradicted, color: "var(--con)" },
        { label: "Unverifiable", value: stats.counts.unverifiable, color: "var(--unv)" },
      ]
    : [];

  return (
    <div className={`ax-root ${collapsed ? "collapsed" : ""}`}>
      <aside className="ax-side">
        <div className="ax-brand">
          <span className="ax-brand-icon">
            <IconLens size={20} />
          </span>
          <span className="ax-brand-name">TrustLens</span>
        </div>
        <button type="button" className="ax-collapse" aria-label="Toggle sidebar" onClick={() => setCollapsed(!collapsed)}>
          ‹
        </button>
        <nav className="ax-nav">
          {nav.map(([p, label, icon]) => (
            <button key={p} type="button" className={page === p ? "active" : ""} onClick={() => setPage(p)}>
              {icon}
              <span>{label}</span>
            </button>
          ))}
        </nav>
        {isMock && (
          <div className="ax-upgrade">
            <div className="ax-upgrade-title">Go live</div>
            <p>Add your Gemini and Groq keys to backend/.env to replace the mock verifiers.</p>
            <div className="ax-orb" aria-hidden="true" />
          </div>
        )}
      </aside>

      <main className="ax-main">
        <header className="ax-head">
          <div>
            <h1>{nav.find((n) => n[0] === page)?.[1]}</h1>
            <p>Check how well an AI answer holds up against real evidence</p>
          </div>
          <form className="ax-ask" onSubmit={submit}>
            <IconSearch />
            <input
              id="question"
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              placeholder="Ask a question…"
              aria-label="Ask a question and TrustLens checks the AI's answer claim by claim"
              autoComplete="off"
            />
            <button type="submit" disabled={loading || !question.trim()}>
              {loading ? "Checking…" : "Check"}
            </button>
          </form>
        </header>

        {error && (
          <p className="ax-error" role="alert">
            {error}
          </p>
        )}

        {page === "overview" && (
          <>
            <section className="ax-stats">
              <StatCard
                icon={<IconShield />}
                label="Trust Score"
                value={stats ? percent(stats.trust) : "—"}
                sub={stats ? `${stats.counts.supported} supported · ${stats.counts.contradicted} contradicted` : "Run a check to begin"}
                series={running}
                tone="violet"
              />
              <StatCard icon={<IconLayers />} label="Claims Checked" value={stats ? String(rows.length) : "—"} sub={stats ? `${stats.review} sent to human review` : "—"} series={gemini} tone="green" />
              <StatCard
                icon={<IconPulse />}
                label="Verifier Agreement"
                value={stats ? percent(stats.agreement) : "—"}
                sub={stats ? `Gemini ${percent(stats.avgConfidence.gemini ?? 0)} · Groq ${percent(stats.avgConfidence.groq_qwen ?? 0)} avg` : "—"}
                series={groq}
                tone="blue"
              />
              <StatCard icon={<IconBook />} label="Evidence Passages" value={stats ? String(stats.evidenceCount) : "—"} sub={stats ? `from ${stats.sourceCount} sources` : "—"} series={evidenceScores} tone="amber" />
            </section>

            <section className="ax-insight">
              <span className="ax-insight-icon">
                <IconBulb />
              </span>
              <div>
                <div className="ax-insight-title">
                  Insights <span>{isMock ? "DEMO" : "LIVE"}</span>
                </div>
                <p>{stats ? stats.insights[0] : "Insights appear here after a check: contradictions, disagreements and claims with no evidence."}</p>
              </div>
              <button type="button" className="ax-link" onClick={() => setPage("claims")}>
                View claims <IconArrowRight />
              </button>
            </section>

            <section className="ax-charts">
              <div className="ax-card">
                <div className="ax-card-head">
                  <div>
                    <h3>Verifier Confidence</h3>
                    <p>Per claim, both verifiers</p>
                  </div>
                  <div className="ax-keys">
                    <span>
                      <i style={{ background: "#8b7bff" }} /> Gemini
                    </span>
                    <span>
                      <i style={{ background: "#3ddc97" }} /> Groq
                    </span>
                  </div>
                </div>
                <div className="ax-big">{stats ? percent(((stats.avgConfidence.gemini ?? 0) + (stats.avgConfidence.groq_qwen ?? 0)) / 2) : "—"}</div>
                <div className="ax-big-sub">average confidence</div>
                {rows.length ? (
                  <LineChart
                    series={[
                      { label: "Gemini", values: gemini, color: "#8b7bff" },
                      { label: "Groq", values: groq, color: "#3ddc97" },
                    ]}
                    labels={labels}
                  />
                ) : (
                  <div className="ax-empty">No data yet</div>
                )}
              </div>

              <div className="ax-card">
                <div className="ax-card-head">
                  <div>
                    <h3>Verdict Breakdown</h3>
                    <p>Share of all claims</p>
                  </div>
                </div>
                <div className="ax-donut-wrap">
                  <Donut segments={donut} size={190} thickness={20}>
                    <b>{stats ? percent(stats.trust) : "—"}</b>
                    <span>trust</span>
                  </Donut>
                  <ul className="ax-legend">
                    {(donut.length ? donut : [
                      { label: "Supported", value: 0, color: "var(--sup)" },
                      { label: "Contradicted", value: 0, color: "var(--con)" },
                      { label: "Unverifiable", value: 0, color: "var(--unv)" },
                    ]).map((s) => (
                      <li key={s.label}>
                        <i style={{ background: s.color }} />
                        <div>
                          <b>{s.label}</b>
                          <span>
                            {s.value} claim{s.value === 1 ? "" : "s"}
                            {rows.length ? ` (${Math.round((s.value / rows.length) * 100)}%)` : ""}
                          </span>
                        </div>
                      </li>
                    ))}
                  </ul>
                </div>
                <button type="button" className="ax-link center" onClick={() => setPage("claims")}>
                  View every claim <IconArrowRight />
                </button>
              </div>

              <div className="ax-card">
                <div className="ax-card-head">
                  <div>
                    <h3>Evidence Match</h3>
                    <p>How closely each passage matches its claim</p>
                  </div>
                </div>
                <div className="ax-big">{evidenceScores.length ? Math.max(...evidenceScores).toFixed(2) : "—"}</div>
                <div className="ax-big-sub">best highlight score</div>
                {evidenceScores.length ? <BarChart values={evidenceScores} labels={evidenceLabels} color="#5b8cff" /> : <div className="ax-empty">No data yet</div>}
                <button type="button" className="ax-link center" onClick={() => setPage("evidence")}>
                  View evidence <IconArrowRight />
                </button>
              </div>
            </section>

            <section className="ax-bottom">
              <div className="ax-card ax-table-card">
                <div className="ax-card-head">
                  <div>
                    <h3>Claims &amp; Verdicts</h3>
                    <p>Both verifiers and the final decision for each claim</p>
                  </div>
                  <button type="button" className="ax-outline" onClick={() => setPage("claims")}>
                    <IconChart /> Details
                  </button>
                </div>
                {rows.length === 0 ? (
                  <div className="ax-empty">Run a check to fill this table.</div>
                ) : (
                  <table className="ax-table">
                    <thead>
                      <tr>
                        <th>Claim</th>
                        <th>Gemini</th>
                        <th>Groq</th>
                        <th>Confidence</th>
                        <th>Final</th>
                        <th>Action</th>
                      </tr>
                    </thead>
                    <tbody>
                      {rows.map((r) => (
                        <tr key={r.claim.claim_id} className={picked === r.claim.claim_id ? "picked" : ""} onClick={() => setPicked(picked === r.claim.claim_id ? null : r.claim.claim_id)}>
                          <td>
                            <div className="ax-claim">
                              <span className={`ax-num n-${r.decision.final_verdict}`}>{claimNumber(r.index)}</span>
                              <span className="ax-claim-text">{r.claim.text}</span>
                            </div>
                          </td>
                          <td>{verdictChip(r.verdictOf.gemini)}</td>
                          <td>{verdictChip(r.verdictOf.groq_qwen)}</td>
                          <td className="num">{percent(((r.confidenceOf.gemini ?? 0) + (r.confidenceOf.groq_qwen ?? 0)) / 2)}</td>
                          <td>{verdictChip(r.decision.final_verdict)}</td>
                          <td className="muted">{ACTION_LABEL[r.action]}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                )}
                {pickedRow && (
                  <ol className="ax-detail">
                    <ClaimCard claim={pickedRow.claim} decision={pickedRow.decision} evidence={pickedRow.evidence} index={pickedRow.index} defaultOpen />
                  </ol>
                )}
              </div>

              <div className="ax-card">
                <div className="ax-card-head">
                  <div>
                    <h3>Recent Activity</h3>
                    <p>Latest verifier decisions</p>
                  </div>
                </div>
                {rows.length === 0 ? (
                  <div className="ax-empty">Nothing yet.</div>
                ) : (
                  <ul className="ax-activity">
                    {rows.slice(0, 6).map((r) => (
                      <li key={r.claim.claim_id}>
                        <span className={`ax-dot n-${r.decision.final_verdict}`}>
                          <IconCheck />
                        </span>
                        <div className="ax-act-body">
                          <b>Claim {claimNumber(r.index)}</b>
                          <span>{r.claim.text}</span>
                        </div>
                        <span className={`ax-act-verdict t-${r.decision.final_verdict}`}>{VERDICT_LABEL[r.decision.final_verdict]}</span>
                      </li>
                    ))}
                  </ul>
                )}
              </div>
            </section>

            {result && (
              <section className="ax-card ax-answer">
                <div className="ax-card-head">
                  <div>
                    <h3>Audited Answer</h3>
                    <p>{result.question}</p>
                  </div>
                </div>
                <AnswerView analysis={result} />
                <p className="ax-fine">{result.trust_score.disclaimer}</p>
              </section>
            )}
          </>
        )}

        {page === "claims" && (
          <section className="ax-card ax-pad">
            {rows.length === 0 ? (
              <div className="ax-empty">Run a check to see every claim.</div>
            ) : (
              <ol className="claims">
                {rows.map((r) => (
                  <ClaimCard key={r.claim.claim_id} claim={r.claim} decision={r.decision} evidence={r.evidence} index={r.index} defaultOpen={r.decision.final_verdict === "contradicted"} />
                ))}
              </ol>
            )}
          </section>
        )}

        {page === "evidence" && (
          <section className="ax-card ax-pad">
            {rows.length === 0 ? (
              <div className="ax-empty">Run a check to see the evidence.</div>
            ) : (
              rows.map((r) => (
                <div key={r.claim.claim_id} className="ax-ev">
                  <h3>
                    <span className={`ax-num n-${r.decision.final_verdict}`}>{claimNumber(r.index)}</span> {r.claim.text}
                  </h3>
                  <EvidencePanel evidence={r.evidence} verifierResults={r.decision.verifier_results} />
                </div>
              ))
            )}
          </section>
        )}

        {page === "history" && (
          <section className="ax-card ax-pad">
            {!history ? (
              <div className="ax-empty">Loading history…</div>
            ) : history.length === 0 ? (
              <div className="ax-empty">No analyses yet.</div>
            ) : (
              <ul className="ax-activity">
                {history.map((h) => (
                  <li key={h.answer_id} className="clickable" onClick={() => { void open(h.answer_id); setPage("overview"); }}>
                    <span className="ax-dot n-supported">
                      <IconClock />
                    </span>
                    <div className="ax-act-body">
                      <b>{h.question ?? "(pasted answer)"}</b>
                      <span>
                        {h.claim_count} claims{h.created_at ? ` · ${new Date(h.created_at).toLocaleString()}` : ""}
                      </span>
                    </div>
                    <span className="ax-act-verdict">{percent(h.trust_score)}</span>
                  </li>
                ))}
              </ul>
            )}
          </section>
        )}

        {page === "settings" && (
          <section className="ax-card ax-pad ax-settings">
            <h3>Appearance</h3>
            <UiSwitch ui={ui} onUi={onUi} theme={theme} onTheme={onTheme} className="ax-switch" />
            <h3>Data source</h3>
            <p className="muted">{isMock ? "Demo mode: the sample analysis and offline mock verifiers. Connect the backend and API keys to go live." : "Connected to the TrustLens backend."}</p>
          </section>
        )}

        {page !== "settings" && (
          <div className="ax-footer">
            <UiSwitch ui={ui} onUi={onUi} theme={theme} onTheme={onTheme} className="ax-switch" />
          </div>
        )}
      </main>
    </div>
  );
}
