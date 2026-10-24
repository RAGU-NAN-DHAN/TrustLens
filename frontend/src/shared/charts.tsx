import { useId, type ReactNode } from "react";

interface SparkProps {
  values: number[];
  color: string;
  height?: number;
  fill?: boolean;
}

function points(values: number[], w: number, h: number, pad = 2): [number, number][] {
  if (values.length === 0) return [];
  const max = Math.max(...values, 1e-6);
  const min = Math.min(...values, 0);
  const span = max - min || 1;
  const step = values.length > 1 ? (w - pad * 2) / (values.length - 1) : 0;
  return values.map((v, i) => [pad + i * step, h - pad - ((v - min) / span) * (h - pad * 2)]);
}

function smoothPath(pts: [number, number][]): string {
  if (pts.length === 0) return "";
  if (pts.length === 1) return `M${pts[0][0]},${pts[0][1]}`;
  let d = `M${pts[0][0]},${pts[0][1]}`;
  for (let i = 1; i < pts.length; i++) {
    const [x0, y0] = pts[i - 1];
    const [x1, y1] = pts[i];
    const cx = (x0 + x1) / 2;
    d += ` C${cx},${y0} ${cx},${y1} ${x1},${y1}`;
  }
  return d;
}

export function Sparkline({ values, color, height = 48, fill = true }: SparkProps) {
  const id = useId();
  const w = 200;
  const pts = points(values, w, height);
  const line = smoothPath(pts);
  const area = pts.length ? `${line} L${pts[pts.length - 1][0]},${height} L${pts[0][0]},${height} Z` : "";
  return (
    <svg className="spark" viewBox={`0 0 ${w} ${height}`} preserveAspectRatio="none" aria-hidden="true">
      <defs>
        <linearGradient id={id} x1="0" x2="0" y1="0" y2="1">
          <stop offset="0" stopColor={color} stopOpacity="0.35" />
          <stop offset="1" stopColor={color} stopOpacity="0" />
        </linearGradient>
      </defs>
      {fill && area && <path d={area} fill={`url(#${id})`} />}
      {line && <path d={line} fill="none" stroke={color} strokeWidth="1.6" vectorEffect="non-scaling-stroke" />}
    </svg>
  );
}

export function MiniBars({ values, color, height = 40 }: { values: number[]; color: string; height?: number }) {
  const n = Math.max(values.length, 1);
  const gap = 3;
  const w = n * 7 + (n - 1) * gap;
  const max = Math.max(...values, 1e-6);
  return (
    <svg className="minibars" viewBox={`0 0 ${w} ${height}`} width={w} height={height} aria-hidden="true">
      {values.map((v, i) => {
        const h = Math.max(3, (v / max) * height);
        return <rect key={i} x={i * (7 + gap)} y={height - h} width="7" height={h} rx="1.5" fill={color} opacity={0.55 + 0.45 * (v / max)} />;
      })}
    </svg>
  );
}

export interface DonutSegment {
  label: string;
  value: number;
  color: string;
}

export function Donut({ segments, size = 180, thickness = 18, children }: { segments: DonutSegment[]; size?: number; thickness?: number; children?: ReactNode }) {
  const total = segments.reduce((s, x) => s + x.value, 0);
  const r = (size - thickness) / 2;
  const c = 2 * Math.PI * r;
  let offset = 0;
  return (
    <div className="donut" style={{ width: size, height: size }}>
      <svg viewBox={`0 0 ${size} ${size}`} width={size} height={size} aria-hidden="true">
        <circle cx={size / 2} cy={size / 2} r={r} fill="none" stroke="rgba(255,255,255,0.06)" strokeWidth={thickness} />
        {total > 0 &&
          segments.map((s) => {
            const len = (s.value / total) * c;
            const gap = segments.filter((x) => x.value > 0).length > 1 ? 3 : 0;
            const el = (
              <circle
                key={s.label}
                cx={size / 2}
                cy={size / 2}
                r={r}
                fill="none"
                stroke={s.color}
                strokeWidth={thickness}
                strokeDasharray={`${Math.max(0, len - gap)} ${c}`}
                strokeDashoffset={-offset}
                transform={`rotate(-90 ${size / 2} ${size / 2})`}
              />
            );
            offset += len;
            return el;
          })}
      </svg>
      <div className="donut-center">{children}</div>
    </div>
  );
}

interface Series {
  label: string;
  values: number[];
  color: string;
}

export function LineChart({ series, labels, height = 180, yMax = 1, yFormat = (v: number) => `${Math.round(v * 100)}%` }: { series: Series[]; labels: string[]; height?: number; yMax?: number; yFormat?: (v: number) => string }) {
  const ids = [useId(), useId(), useId()];
  const w = 520;
  const left = 38;
  const bottom = 22;
  const plotH = height - bottom - 6;
  const plotW = w - left - 8;
  const n = Math.max(labels.length, 1);
  const x = (i: number) => left + (n > 1 ? (i / (n - 1)) * plotW : plotW / 2);
  const y = (v: number) => 6 + plotH - (Math.min(v, yMax) / yMax) * plotH;
  const ticks = [0, 0.25, 0.5, 0.75, 1].map((t) => t * yMax);
  return (
    <svg className="linechart" viewBox={`0 0 ${w} ${height}`} role="img" aria-label="line chart">
      {ticks.map((t) => (
        <g key={t}>
          <line x1={left} x2={w - 8} y1={y(t)} y2={y(t)} stroke="rgba(255,255,255,0.06)" />
          <text x={left - 8} y={y(t) + 4} textAnchor="end" className="axis">
            {yFormat(t)}
          </text>
        </g>
      ))}
      {labels.map((l, i) => (
        <text key={l + i} x={x(i)} y={height - 4} textAnchor="middle" className="axis">
          {l}
        </text>
      ))}
      {series.map((s, si) => {
        const pts = s.values.map((v, i) => [x(i), y(v)] as [number, number]);
        const line = smoothPath(pts);
        const area = pts.length ? `${line} L${pts[pts.length - 1][0]},${y(0)} L${pts[0][0]},${y(0)} Z` : "";
        return (
          <g key={s.label}>
            <defs>
              <linearGradient id={ids[si % 3]} x1="0" x2="0" y1="0" y2="1">
                <stop offset="0" stopColor={s.color} stopOpacity="0.28" />
                <stop offset="1" stopColor={s.color} stopOpacity="0" />
              </linearGradient>
            </defs>
            {area && <path d={area} fill={`url(#${ids[si % 3]})`} />}
            <path d={line} fill="none" stroke={s.color} strokeWidth="2" />
            {pts.map(([px, py], i) => (
              <circle key={i} cx={px} cy={py} r="3.2" fill="#0b1020" stroke={s.color} strokeWidth="1.6">
                <title>{`${s.label} · ${labels[i]}: ${yFormat(s.values[i])}`}</title>
              </circle>
            ))}
          </g>
        );
      })}
    </svg>
  );
}

export function BarChart({ values, labels, color, height = 180 }: { values: number[]; labels: string[]; color: string; height?: number }) {
  const w = 520;
  const left = 38;
  const bottom = 22;
  const plotH = height - bottom - 6;
  const plotW = w - left - 8;
  const n = Math.max(values.length, 1);
  const slot = plotW / n;
  const bw = Math.min(26, slot * 0.6);
  const y = (v: number) => 6 + plotH - Math.min(v, 1) * plotH;
  return (
    <svg className="barchart" viewBox={`0 0 ${w} ${height}`} role="img" aria-label="bar chart">
      {[0, 0.5, 1].map((t) => (
        <g key={t}>
          <line x1={left} x2={w - 8} y1={y(t)} y2={y(t)} stroke="rgba(255,255,255,0.06)" />
          <text x={left - 8} y={y(t) + 4} textAnchor="end" className="axis">
            {t.toFixed(1)}
          </text>
        </g>
      ))}
      {values.map((v, i) => (
        <g key={i}>
          <rect x={left + i * slot + (slot - bw) / 2} y={y(v)} width={bw} height={y(0) - y(v)} rx="3" fill={color}>
            <title>{`${labels[i]}: ${v.toFixed(2)}`}</title>
          </rect>
          <text x={left + i * slot + slot / 2} y={height - 4} textAnchor="middle" className="axis">
            {labels[i]}
          </text>
        </g>
      ))}
    </svg>
  );
}
